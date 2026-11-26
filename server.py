"""Face Access server: recognises staff faces sent from the web page."""
import base64
import binascii
import calendar
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import psycopg
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import date, datetime, time as datetime_time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import database as attendance_db
import chatbot_service
import storage as face_storage

# ---------- Settings ----------
DET_MODEL = "models/face_detection_yunet_2023mar.onnx"
REC_MODEL = "models/face_recognition_sface_2021dec.onnx"
LIVENESS_MODEL = "models/face_antispoof_98.20.onnx"
LIVENESS_MODEL_SHA256 = "af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6"
LIVENESS_THRESHOLD = 0.5
LIVENESS_INPUT_SIZE = 128
LIVENESS_CROP_PADDING = 1.5
STAFF_DIR = "staff"
LOG_FILE = "entries.csv"
ADMIN_FILE = "admin.json"
SETTINGS_FILE = "settings.json"
THRESHOLD = 0.40
VIDEO_STEP = 10
MAX_VIDEO_SAMPLES = 30
IMG_EXT = (".jpg", ".jpeg", ".png")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ATTENDANCE_PHOTO_BYTES = 5 * 1024 * 1024
MAX_ATTENDANCE_PHOTO_DATA_LENGTH = 4 * ((MAX_ATTENDANCE_PHOTO_BYTES + 2) // 3)
ATTENDANCE_PHOTO_DIR = os.path.abspath(
    os.environ.get(
        "FACE_ACCESS_PHOTO_DIR",
        os.path.join(os.path.dirname(__file__), "attendance_photos"),
    )
)
CORRECTION_DOCUMENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "correction_documents")
)
ACCESS_COOLDOWN_SECONDS = 30
LOG_FIELDS = ["time", "event", "status", "name", "score", "method", "details"]
SESSION_COOKIE = "face_access_admin"
EMPLOYEE_SESSION_COOKIE = "face_access_employee"
HOD_SESSION_COOKIE = "face_access_hod"
SESSION_SECONDS = 30 * 60
PBKDF2_ITERATIONS = 300_000
EMPLOYEE_PASSWORD_MIN_LENGTH = 10
SPEAK_PS = ("Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:FACE_SAY)")

# ---------- Models ----------
detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), 0.7, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")
liveness_model = None
lock = threading.Lock()
log_lock = threading.Lock()
admin_lock = threading.Lock()
employee_session_lock = threading.Lock()
hod_session_lock = threading.Lock()
STAFF = {}
STAFF_META = {}
ADMIN_SESSIONS = {}
RECOGNITION_PROOFS = {}
ACCESS_COOLDOWNS = {}
LOGIN_FAILURES = {}
LOGIN_LOCKOUTS = {}
EMPLOYEE_SESSIONS = {}
EMPLOYEE_LOGIN_FAILURES = {}
EMPLOYEE_LOGIN_LOCKOUTS = {}
HOD_SESSIONS = {}
HOD_LOGIN_FAILURES = {}
HOD_LOGIN_LOCKOUTS = {}


def detect(img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    return faces if faces is not None else []


def embed(img, face):
    aligned = recognizer.alignCrop(img, face)
    feat = recognizer.feature(aligned).flatten().astype(np.float32)
    return feat / np.linalg.norm(feat)


class LivenessUnavailableError(RuntimeError):
    pass


def load_liveness_model():
    try:
        with open(LIVENESS_MODEL, "rb") as model_file:
            model_bytes = model_file.read()
    except OSError as exc:
        print(f"Liveness verification unavailable: cannot read {LIVENESS_MODEL}: {exc}")
        return None
    if hashlib.sha256(model_bytes).hexdigest() != LIVENESS_MODEL_SHA256:
        print(f"Liveness verification unavailable: {LIVENESS_MODEL} checksum mismatch.")
        return None
    try:
        return cv2.dnn.readNetFromONNX(LIVENESS_MODEL)
    except cv2.error as exc:
        print(f"Liveness verification unavailable: could not load model: {exc}")
        return None


def assess_liveness(img, face):
    if liveness_model is None:
        raise LivenessUnavailableError("The anti-spoof model is not available.")

    x, y, width, height = (float(value) for value in face[:4])
    crop_size = int(round(max(width, height) * LIVENESS_CROP_PADDING))
    if crop_size < 2:
        raise LivenessUnavailableError("The detected face is too small to verify.")
    left = int(round(x + width / 2 - crop_size / 2))
    top = int(round(y + height / 2 - crop_size / 2))
    right, bottom = left + crop_size, top + crop_size
    image_height, image_width = img.shape[:2]
    crop = img[max(0, top):min(image_height, bottom), max(0, left):min(image_width, right)]
    if crop.size == 0:
        raise LivenessUnavailableError("The detected face crop is invalid.")
    crop = cv2.copyMakeBorder(
        crop,
        max(0, -top),
        max(0, bottom - image_height),
        max(0, -left),
        max(0, right - image_width),
        cv2.BORDER_REFLECT_101,
    )
    interpolation = (
        cv2.INTER_LANCZOS4
        if crop_size < LIVENESS_INPUT_SIZE
        else cv2.INTER_AREA
    )
    crop = cv2.resize(
        crop,
        (LIVENESS_INPUT_SIZE, LIVENESS_INPUT_SIZE),
        interpolation=interpolation,
    )
    blob = cv2.dnn.blobFromImage(
        crop, 1 / 255.0, (LIVENESS_INPUT_SIZE, LIVENESS_INPUT_SIZE), swapRB=True
    )
    try:
        liveness_model.setInput(blob)
        logits = np.asarray(liveness_model.forward(), dtype=np.float64).reshape(-1)
    except cv2.error as exc:
        raise LivenessUnavailableError("The anti-spoof model could not process this frame.") from exc
    if logits.size != 2 or not np.isfinite(logits).all():
        raise LivenessUnavailableError("The anti-spoof model returned an invalid result.")
    logit_difference = float(logits[0] - logits[1])
    score = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit_difference))))
    return score


def recognition_liveness(score):
    return score is not None and score >= LIVENESS_THRESHOLD


def recognition_proof_expired(proof, now):
    return proof[2] < now


def load_antispoof_model():
    global liveness_model
    liveness_model = load_liveness_model()
    if liveness_model is not None:
        print("Loaded local face anti-spoof model.")


load_antispoof_model()


def largest_face_embedding(img):
    faces = detect(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    return embed(img, face)


def _supabase_storage_enabled():
    return bool(
        os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


def _staff_storage_prefix(name):
    if not valid_name(name):
        raise ValueError("Invalid employee name for storage path.")
    return name.strip()


def _load_staff_from_storage():
    db = {}
    folders = face_storage.list_objects(face_storage.BUCKET_EMPLOYEE_FACES)
    for folder in sorted(item.get("name", "") for item in folders if item.get("name")):
        samples = []
        prefix = _staff_storage_prefix(folder)
        objects = face_storage.list_objects(face_storage.BUCKET_EMPLOYEE_FACES, prefix)
        for item in sorted(objects, key=lambda entry: entry.get("name", "")):
            filename = item.get("name", "")
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMG_EXT + VID_EXT:
                continue
            object_path = f"{prefix}/{filename}"
            content = face_storage.download_bytes(
                face_storage.BUCKET_EMPLOYEE_FACES, object_path
            )
            if ext in IMG_EXT:
                img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    embedding = largest_face_embedding(img)
                    if embedding is not None:
                        samples.append(embedding)
            else:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp:
                        temp.write(content)
                        temp_path = temp.name
                    cap = cv2.VideoCapture(temp_path)
                    i = taken = 0
                    while taken < MAX_VIDEO_SAMPLES:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if i % VIDEO_STEP == 0:
                            embedding = largest_face_embedding(frame)
                            if embedding is not None:
                                samples.append(embedding)
                                taken += 1
                        i += 1
                    cap.release()
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
        if samples:
            db[prefix] = np.stack(samples)
            print(f"Loaded {prefix}: {len(samples)} face samples from Supabase Storage")
        else:
            print(f"WARNING: no face found for {prefix}")
    return db


def load_staff():
    if _supabase_storage_enabled():
        return _load_staff_from_storage()
    db = {}
    os.makedirs(STAFF_DIR, exist_ok=True)
    for name in sorted(os.listdir(STAFF_DIR)):
        folder = os.path.join(STAFF_DIR, name)
        if not os.path.isdir(folder) or os.path.islink(folder):
            continue
        samples = []
        for file in sorted(os.listdir(folder)):
            path = os.path.join(folder, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in IMG_EXT:
                img = cv2.imread(path)
                if img is not None:
                    embedding = largest_face_embedding(img)
                    if embedding is not None:
                        samples.append(embedding)
            elif ext in VID_EXT:
                cap = cv2.VideoCapture(path)
                i = taken = 0
                while taken < MAX_VIDEO_SAMPLES:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if i % VIDEO_STEP == 0:
                        embedding = largest_face_embedding(frame)
                        if embedding is not None:
                            samples.append(embedding)
                            taken += 1
                    i += 1
                cap.release()
        if samples:
            db[name] = np.stack(samples)
            print(f"Loaded {name}: {len(samples)} face samples")
        else:
            print(f"WARNING: no face found for {name}")
    return db


def identify(feat):
    best_name, best_score = "Unknown", 0.0
    for name, matrix in STAFF.items():
        score = float(np.max(matrix @ feat))
        if score > best_score:
            best_name, best_score = name, score
    if best_score < THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


def valid_name(name):
    return (
        isinstance(name, str)
        and bool(name.strip())
        and len(name.strip()) <= 80
        and name.strip() not in {".", ".."}
        and all(char.isalnum() or char in " _-.'" for char in name.strip())
    )


def default_meta():
    return {"active": True, "start": "00:00", "end": "23:59"}


def load_meta():
    if os.environ.get("DATABASE_URL"):
        raw = attendance_db.settings().get("staff_meta", "{}")
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Stored staff settings must contain valid JSON.") from exc
    else:
        if not os.path.exists(SETTINGS_FILE):
            return {}
        with open(SETTINGS_FILE, encoding="utf-8") as file:
            data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Staff settings must contain a JSON object.")
    result = {}
    for name, settings in data.items():
        if not valid_name(name) or not isinstance(settings, dict):
            continue
        meta = default_meta()
        meta["active"] = bool(settings.get("active", True))
        for key in ("start", "end"):
            value = settings.get(key, meta[key])
            if isinstance(value, str) and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                meta[key] = value
        result[name] = meta
    return result


def save_meta():
    if os.environ.get("DATABASE_URL"):
        attendance_db.update_settings({
            "staff_meta": json.dumps(STAFF_META, separators=(",", ":"))
        })
        return
    fd, temp_path = tempfile.mkstemp(prefix=".settings-", dir=".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(STAFF_META, file, indent=2)
            file.write("\n")
        os.replace(temp_path, SETTINGS_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def access_reason(name, now=None):
    if name not in STAFF:
        return "unknown"
    meta = STAFF_META.get(name, default_meta())
    if not meta.get("active", True):
        return "inactive"
    now = now or datetime.now()
    current = now.hour * 60 + now.minute
    start_hour, start_minute = map(int, meta["start"].split(":"))
    end_hour, end_minute = map(int, meta["end"].split(":"))
    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    within = start <= current <= end if start <= end else current >= start or current <= end
    return None if within else "outside allowed hours"


def append_entry(event, status, name, score="", method="", details=""):
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "status": status,
        "name": name,
        "score": f"{score:.3f}" if isinstance(score, (float, int)) else score,
        "method": method,
        "details": details,
    }
    with log_lock:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE):
            with open(LOG_FILE, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != LOG_FIELDS:
                    old_rows = list(reader)
                    fd, temp_path = tempfile.mkstemp(prefix=".entries-", dir=".")
                    try:
                        with os.fdopen(fd, "w", newline="", encoding="utf-8") as temp_file:
                            writer = csv.DictWriter(temp_file, fieldnames=LOG_FIELDS)
                            writer.writeheader()
                            for old_row in old_rows:
                                normalized = {key: old_row.get(key, "") or "" for key in LOG_FIELDS}
                                normalized["event"] = normalized["event"] or "entry"
                                normalized["status"] = normalized["status"] or "granted"
                                normalized["method"] = normalized["method"] or "face"
                                writer.writerow(normalized)
                        os.replace(temp_path, LOG_FILE)
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
    attendance_db.log_access_event(row)


def _is_admin_authenticated(request):
    token = request.cookies.get(SESSION_COOKIE, "")
    with admin_lock:
        expires = ADMIN_SESSIONS.get(token)
        if not token or not expires or expires < time.time():
            ADMIN_SESSIONS.pop(token, None)
            return False
        return True


def ensure_admin(request):
    if not _is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Admin login required.")


def admin_record():
    if os.environ.get("DATABASE_URL"):
        raw = attendance_db.settings().get("admin_pin_record", "")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Stored admin PIN data is invalid.") from exc
    else:
        if not os.path.exists(ADMIN_FILE):
            return None
        with open(ADMIN_FILE, encoding="utf-8") as file:
            data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("salt"), str) or not isinstance(data.get("hash"), str):
        raise ValueError("Admin PIN data is invalid.")
    return data


def set_admin_pin(pin):
    salt = secrets.token_bytes(16)
    pin_hash = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    if os.environ.get("DATABASE_URL"):
        attendance_db.update_settings({
            "admin_pin_record": json.dumps({
                "salt": salt.hex(),
                "hash": pin_hash.hex(),
            }, separators=(",", ":"))
        })
        return
    fd, temp_path = tempfile.mkstemp(prefix=".admin-", dir=".")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump({"salt": salt.hex(), "hash": pin_hash.hex()}, file)
            file.write("\n")
        os.replace(temp_path, ADMIN_FILE)
        os.chmod(ADMIN_FILE, 0o600)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def set_session(response):
    token = secrets.token_urlsafe(32)
    with admin_lock:
        ADMIN_SESSIONS[token] = time.time() + SESSION_SECONDS
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )


def clear_employee_session(request, response):
    token = request.cookies.get(EMPLOYEE_SESSION_COOKIE, "")
    with employee_session_lock:
        EMPLOYEE_SESSIONS.pop(token, None)
    response.delete_cookie(
        EMPLOYEE_SESSION_COOKIE, path="/", httponly=True, samesite="strict"
    )


def revoke_employee_sessions(employee_id):
    with employee_session_lock:
        expired = [
            token for token, session in EMPLOYEE_SESSIONS.items()
            if session["employee_id"] == employee_id
        ]
        for token in expired:
            EMPLOYEE_SESSIONS.pop(token, None)


def set_employee_session(employee_id, response, secure=False):
    token = secrets.token_urlsafe(32)
    with employee_session_lock:
        EMPLOYEE_SESSIONS[token] = {
            "employee_id": employee_id,
            "expires": time.time() + SESSION_SECONDS,
        }
    response.set_cookie(
        EMPLOYEE_SESSION_COOKIE,
        token,
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="strict",
        secure=secure,
        path="/",
    )


def _employee_from_request(request):
    token = request.cookies.get(EMPLOYEE_SESSION_COOKIE, "")
    with employee_session_lock:
        session = EMPLOYEE_SESSIONS.get(token)
        if not token or session is None or session["expires"] < time.time():
            EMPLOYEE_SESSIONS.pop(token, None)
            return None
        employee_id = session["employee_id"]
    employee = attendance_db.employee_by_id(employee_id)
    account = attendance_db.employee_account(employee_id)
    if employee is None or account is None or not account["is_active"] or employee["status"] != "active":
        revoke_employee_sessions(employee_id)
        return None
    with employee_session_lock:
        if token in EMPLOYEE_SESSIONS:
            EMPLOYEE_SESSIONS[token]["expires"] = time.time() + SESSION_SECONDS
    return employee


def ensure_employee(request):
    employee = _employee_from_request(request)
    if employee is None:
        raise HTTPException(status_code=401, detail="Employee login required.")
    return employee


def check_employee_login_rate(request):
    client = request.client.host if request.client else "unknown"
    now = time.time()
    if EMPLOYEE_LOGIN_LOCKOUTS.get(client, 0) > now:
        raise HTTPException(status_code=429, detail="Too many incorrect login attempts. Try again later.")
    return client


def set_employee_login_failure(client):
    failures = EMPLOYEE_LOGIN_FAILURES.get(client, 0) + 1
    EMPLOYEE_LOGIN_FAILURES[client] = failures
    if failures >= 5:
        EMPLOYEE_LOGIN_LOCKOUTS[client] = time.time() + 60
        EMPLOYEE_LOGIN_FAILURES[client] = 0


def employee_password_hash(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return salt.hex(), digest.hex()


def employee_password_matches(password, salt_hex, expected_hex):
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hex)
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail="Employee account credentials are invalid.") from None
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(actual, expected)


def check_hod_login_rate(request):
    client = request.client.host if request.client else "unknown"
    now = time.time()
    if HOD_LOGIN_LOCKOUTS.get(client, 0) > now:
        raise HTTPException(status_code=429, detail="Too many incorrect login attempts. Try again later.")
    return client


def set_hod_login_failure(client):
    failures = HOD_LOGIN_FAILURES.get(client, 0) + 1
    HOD_LOGIN_FAILURES[client] = failures
    if failures >= 5:
        HOD_LOGIN_LOCKOUTS[client] = time.time() + 60
        HOD_LOGIN_FAILURES[client] = 0


def revoke_hod_sessions(hod_id):
    with hod_session_lock:
        for token, session in list(HOD_SESSIONS.items()):
            if session["hod_id"] == hod_id:
                HOD_SESSIONS.pop(token, None)


def set_hod_session(hod_id, response, secure=False):
    token = secrets.token_urlsafe(32)
    with hod_session_lock:
        HOD_SESSIONS[token] = {"hod_id": hod_id, "expires": time.time() + SESSION_SECONDS}
    response.set_cookie(
        HOD_SESSION_COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
        samesite="strict", secure=secure, path="/",
    )


def _hod_from_request(request):
    token = request.cookies.get(HOD_SESSION_COOKIE, "")
    with hod_session_lock:
        session = HOD_SESSIONS.get(token)
        if not token or session is None or session["expires"] < time.time():
            HOD_SESSIONS.pop(token, None)
            return None
        hod_id = session["hod_id"]
    hod = attendance_db.hod_account_by_id(hod_id)
    if hod is None or not hod["is_active"]:
        revoke_hod_sessions(hod_id)
        return None
    with hod_session_lock:
        if token in HOD_SESSIONS:
            HOD_SESSIONS[token]["expires"] = time.time() + SESSION_SECONDS
    return hod


def ensure_hod(request):
    hod = _hod_from_request(request)
    if hod is None:
        raise HTTPException(status_code=401, detail="HOD login required.")
    return hod


def check_login_rate(request):
    client = request.client.host if request.client else "unknown"
    now = time.time()
    if LOGIN_LOCKOUTS.get(client, 0) > now:
        raise HTTPException(status_code=429, detail="Too many incorrect PIN attempts. Try again later.")
    return client


def set_login_failure(client):
    failures = LOGIN_FAILURES.get(client, 0) + 1
    LOGIN_FAILURES[client] = failures
    if failures >= 5:
        LOGIN_LOCKOUTS[client] = time.time() + 60
        LOGIN_FAILURES[client] = 0


def query_entries(name="", start_date="", end_date="", status="", event=""):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline="", encoding="utf-8") as file:
        entries = list(csv.DictReader(file))
    results = []
    for entry in entries:
        timestamp = entry.get("time", "")
        try:
            entry_date = datetime.fromisoformat(timestamp).date()
        except ValueError:
            continue
        if start_date and entry_date < date.fromisoformat(start_date):
            continue
        if end_date and entry_date > date.fromisoformat(end_date):
            continue
        if name and name.casefold() not in entry.get("name", "").casefold():
            continue
        entry_status = entry.get("status") or "granted"
        entry_event = entry.get("event") or "entry"
        if status and entry_status != status:
            continue
        if event and entry_event != event:
            continue
        results.append({
            "time": timestamp,
            "event": entry_event,
            "status": entry_status,
            "name": entry.get("name", ""),
            "score": entry.get("score", ""),
            "method": entry.get("method") or "face",
            "details": entry.get("details", ""),
        })
    return results


def load_recent_access_cooldowns(now=None):
    now = time.time() if now is None else now
    cooldowns = {}
    if not os.path.exists(LOG_FILE):
        return cooldowns
    with open(LOG_FILE, newline="", encoding="utf-8") as file:
        for entry in csv.DictReader(file):
            if (
                entry.get("event") != "entry"
                or entry.get("status") != "granted"
                or entry.get("method") != "face"
            ):
                continue
            try:
                expires = datetime.fromisoformat(entry["time"]).timestamp() + ACCESS_COOLDOWN_SECONDS
            except (KeyError, ValueError):
                continue
            if expires > now:
                name = entry.get("name", "")
                cooldowns[name] = max(cooldowns.get(name, 0), expires)
    return cooldowns


def validate_filters(start_date, end_date):
    try:
        if start_date:
            date.fromisoformat(start_date)
        if end_date:
            date.fromisoformat(end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Dates must use YYYY-MM-DD format.") from exc
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be on or before end date.")


def safe_attendance_photo_path(relative_path):
    if not relative_path or os.path.isabs(relative_path):
        raise ValueError("Attendance photo path is invalid.")
    root = os.path.realpath(ATTENDANCE_PHOTO_DIR)
    path = os.path.realpath(os.path.join(root, relative_path))
    if os.path.commonpath((root, path)) != root or os.path.islink(path):
        raise ValueError("Attendance photo path is outside the evidence directory.")
    return path


def decode_attendance_photo(photo_data, expected_name):
    try:
        image_bytes = base64.b64decode(photo_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Attendance photo data is invalid; attendance was not recorded.",
        ) from exc
    if not image_bytes or len(image_bytes) > MAX_ATTENDANCE_PHOTO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Attendance photo must be 5 MB or smaller; attendance was not recorded.",
        )
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise HTTPException(
            status_code=400,
            detail="Attendance photo must be a JPEG image; attendance was not recorded.",
        )
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Attendance photo could not be decoded; attendance was not recorded.",
        )
    if image.shape[0] * image.shape[1] > 40_000_000:
        raise HTTPException(
            status_code=413,
            detail="Attendance photo dimensions are too large; attendance was not recorded.",
        )
    with lock:
        matching_faces = [
            (face, score)
            for face in detect(image)
            for name, score in (identify(embed(image, face)),)
            if name == expected_name and score >= THRESHOLD
        ]
        if matching_faces:
            best_face, best_score = max(matching_faces, key=lambda match: match[1])
            try:
                liveness_score = assess_liveness(image, best_face)
            except LivenessUnavailableError as exc:
                print(f"Liveness verification failed: {exc}")
                raise HTTPException(
                    status_code=503,
                    detail="Liveness verification is unavailable; attendance was not recorded.",
                ) from exc
    if not matching_faces:
        raise HTTPException(
            status_code=403,
            detail=(
                "The submitted camera photo does not show the recognized employee; "
                "attendance was not recorded."
            ),
        )
    if not recognition_liveness(liveness_score):
        raise HTTPException(
            status_code=403,
            detail=(
                "Liveness verification failed for the attendance photo; "
                "attendance was not recorded."
            ),
        )
    return image_bytes, best_score, liveness_score


def save_attendance_photo(action, employee_id, captured_at, image_bytes):
    folder = os.path.join(
        ATTENDANCE_PHOTO_DIR,
        "checkin" if action == "check_in" else "checkout",
        f"{captured_at.year:04d}",
        f"{captured_at.month:02d}",
    )
    os.makedirs(folder, mode=0o700, exist_ok=True)
    if os.name == "posix":
        os.chmod(folder, 0o700)
    action_name = "checkin" if action == "check_in" else "checkout"
    filename = (
        f"{employee_id}_{captured_at.date().isoformat()}_{action_name}_"
        f"{uuid.uuid4().hex}.jpg"
    )
    final_path = os.path.join(folder, filename)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".attendance-photo-", dir=folder)
    try:
        with os.fdopen(descriptor, "wb") as file:
            if os.name == "posix":
                os.fchmod(file.fileno(), 0o600)
            file.write(image_bytes)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, final_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return os.path.relpath(final_path, ATTENDANCE_PHOTO_DIR)


def remove_attendance_photo_file(relative_path):
    if not relative_path:
        return
    path = safe_attendance_photo_path(relative_path)
    if os.path.exists(path):
        os.remove(path)


def attendance_photo_retention_days():
    raw_days = os.environ.get("FACE_ACCESS_PHOTO_RETENTION_DAYS", "0")
    try:
        days = int(raw_days)
    except ValueError as exc:
        raise ValueError(
            "FACE_ACCESS_PHOTO_RETENTION_DAYS must be a non-negative integer."
        ) from exc
    if days < 0:
        raise ValueError("FACE_ACCESS_PHOTO_RETENTION_DAYS must be a non-negative integer.")
    return days


def apply_attendance_photo_retention():
    retention_days = attendance_photo_retention_days()
    if retention_days == 0:
        return
    timezone = ZoneInfo(attendance_db.settings()["timezone"])
    cutoff = datetime.now(timezone) - timedelta(days=retention_days)
    for record in attendance_db.attendance_photo_records():
        for action, timestamp_key, path_key in (
            ("check_in", "checkin_photo_timestamp", "checkin_photo_path"),
            ("check_out", "checkout_photo_timestamp", "checkout_photo_path"),
        ):
            timestamp = record[timestamp_key]
            relative_path = record[path_key]
            if not timestamp or not relative_path or datetime.fromisoformat(timestamp) >= cutoff:
                continue
            path = safe_attendance_photo_path(relative_path)
            if os.path.exists(path):
                os.remove(path)
            attendance_db.clear_attendance_photo(record["attendance_id"], action)
    for movement in attendance_db.attendance_movement_photo_records():
        if not movement["photo_timestamp"]:
            continue
        if datetime.fromisoformat(movement["photo_timestamp"]) >= cutoff:
            continue
        path = safe_attendance_photo_path(movement["photo_path"])
        if os.path.exists(path):
            os.remove(path)
        attendance_db.clear_attendance_movement_photo(movement["movement_id"])


with lock:
    STAFF_META = load_meta()
    STAFF = load_staff()
for staff_name in STAFF:
    STAFF_META.setdefault(staff_name, default_meta())
ACCESS_COOLDOWNS.update(load_recent_access_cooldowns())
attendance_db.initialize_database()
attendance_db.sync_legacy_staff(STAFF, STAFF_META)
attendance_db.import_legacy_entries(LOG_FILE)
os.makedirs(ATTENDANCE_PHOTO_DIR, mode=0o700, exist_ok=True)
if os.name == "posix":
    os.chmod(ATTENDANCE_PHOTO_DIR, 0o700)
apply_attendance_photo_retention()
os.makedirs(CORRECTION_DOCUMENT_DIR, mode=0o700, exist_ok=True)
if os.name == "posix":
    os.chmod(CORRECTION_DOCUMENT_DIR, 0o700)
attendance_db.detect_missing_checkouts(database_path=attendance_db.DATABASE_FILE)
THRESHOLD = float(attendance_db.settings()["face_threshold"])
print(f"Ready: {len(STAFF)} staff loaded. Open http://localhost:8000")


def say(text):
    if sys.platform == "darwin":
        subprocess.Popen(["say", text])
    else:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", SPEAK_PS],
            env={**os.environ, "FACE_SAY": text},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


# ---------- API ----------
app = FastAPI(title="Face Access")


@app.middleware("http")
async def prevent_employee_data_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/employee/") or request.url.path == "/employee":
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/")
def home():
    return FileResponse("web/index.html")


@app.get("/admin/status")
def admin_status(request: Request):
    configured = admin_record() is not None
    token = request.cookies.get(SESSION_COOKIE, "")
    with admin_lock:
        authenticated = bool(ADMIN_SESSIONS.get(token, 0) > time.time())
    return {"pin_configured": configured, "authenticated": authenticated}


class PinPayload(BaseModel):
    pin: str = Field(min_length=6, max_length=128)


class ChangePinPayload(BaseModel):
    current_pin: str = Field(min_length=6, max_length=128)
    new_pin: str = Field(min_length=6, max_length=128)


class EmployeeLoginPayload(BaseModel):
    identifier: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class EmployeeAccountCreate(BaseModel):
    employee_id: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=EMPLOYEE_PASSWORD_MIN_LENGTH, max_length=128)
    annual_leave_days: float | None = Field(default=None, ge=0, le=365)


class EmployeeAccountUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=EMPLOYEE_PASSWORD_MIN_LENGTH, max_length=128)
    is_active: bool | None = None
    annual_leave_days: float | None = Field(default=None, ge=0, le=365)


class EmployeePasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=EMPLOYEE_PASSWORD_MIN_LENGTH, max_length=128)


class HODLoginPayload(BaseModel):
    identifier: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class HODAccountCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    department: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=10, max_length=128)


class HODAccountUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=100)
    email: str | None = Field(default=None, min_length=3, max_length=254)
    department: str | None = Field(default=None, min_length=1, max_length=100)
    password: str | None = Field(default=None, min_length=10, max_length=128)
    is_active: bool | None = None


@app.post("/admin/setup")
def admin_setup(data: PinPayload, request: Request, response: Response):
    check_login_rate(request)
    with admin_lock:
        if admin_record() is not None:
            raise HTTPException(status_code=409, detail="Admin PIN is already configured. Please log in.")
        set_admin_pin(data.pin)
    LOGIN_FAILURES.pop(request.client.host if request.client else "unknown", None)
    clear_employee_session(request, response)
    set_session(response)
    return {"ok": True}


@app.post("/admin/change-pin")
def admin_change_pin(data: ChangePinPayload, request: Request, response: Response):
    ensure_admin(request)
    client = check_login_rate(request)
    record = admin_record()
    if record is None:
        raise HTTPException(status_code=409, detail="Set up an admin PIN first.")
    try:
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Admin PIN data is invalid.") from exc
    actual = hashlib.pbkdf2_hmac(
        "sha256", data.current_pin.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    if not hmac.compare_digest(actual, expected):
        set_login_failure(client)
        raise HTTPException(status_code=401, detail="Current PIN is incorrect.")
    with admin_lock:
        set_admin_pin(data.new_pin)
        ADMIN_SESSIONS.clear()
    LOGIN_FAILURES.pop(client, None)
    LOGIN_LOCKOUTS.pop(client, None)
    clear_employee_session(request, response)
    set_session(response)
    return {"ok": True}


@app.post("/admin/login")
def admin_login(data: PinPayload, request: Request, response: Response):
    client = check_login_rate(request)
    record = admin_record()
    if record is None:
        raise HTTPException(status_code=409, detail="Set up an admin PIN first.")
    try:
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Admin PIN data is invalid.") from exc
    actual = hashlib.pbkdf2_hmac("sha256", data.pin.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    if not hmac.compare_digest(actual, expected):
        set_login_failure(client)
        raise HTTPException(status_code=401, detail="Incorrect PIN.")
    LOGIN_FAILURES.pop(client, None)
    LOGIN_LOCKOUTS.pop(client, None)
    clear_employee_session(request, response)
    set_session(response)
    return {"ok": True}


@app.post("/admin/logout")
def admin_logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE, "")
    with admin_lock:
        ADMIN_SESSIONS.pop(token, None)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return {"ok": True}


@app.get("/employee/auth/status")
def employee_auth_status(request: Request):
    return {"authenticated": _employee_from_request(request) is not None}


@app.post("/employee/auth/login")
def employee_login(data: EmployeeLoginPayload, request: Request, response: Response):
    client = check_employee_login_rate(request)
    account = attendance_db.employee_account_for_login(data.identifier.strip())
    valid = False
    if account is not None:
        valid = employee_password_matches(
            data.password, account["password_salt"], account["password_hash"]
        )
    else:
        hashlib.pbkdf2_hmac(
            "sha256", data.password.encode("utf-8"), bytes(16), PBKDF2_ITERATIONS
        )
    if (
        account is None or not valid or not account["is_active"]
        or account["employee_status"] != "active"
    ):
        set_employee_login_failure(client)
        raise HTTPException(status_code=401, detail="Employee ID/email or password is incorrect.")
    EMPLOYEE_LOGIN_FAILURES.pop(client, None)
    EMPLOYEE_LOGIN_LOCKOUTS.pop(client, None)
    admin_token = request.cookies.get(SESSION_COOKIE, "")
    if admin_token:
        with admin_lock:
            ADMIN_SESSIONS.pop(admin_token, None)
        response.delete_cookie(
            SESSION_COOKIE, path="/", httponly=True, samesite="strict"
        )
    attendance_db.update_employee_last_login(account["employee_id"])
    set_employee_session(
        account["employee_id"], response, secure=request.url.scheme == "https"
    )
    return {"ok": True}


@app.post("/employee/auth/logout")
def employee_logout(request: Request, response: Response):
    clear_employee_session(request, response)
    return {"ok": True}


@app.get("/hod/auth/status")
def hod_auth_status(request: Request):
    hod = _hod_from_request(request)
    return {"authenticated": hod is not None, "department": hod["department"] if hod else None}


@app.post("/hod/auth/login")
def hod_login(data: HODLoginPayload, request: Request, response: Response):
    client = check_hod_login_rate(request)
    account = attendance_db.hod_account_for_login(data.identifier.strip())
    valid = False
    if account is not None:
        valid = employee_password_matches(
            data.password, account["password_salt"], account["password_hash"]
        )
    else:
        hashlib.pbkdf2_hmac("sha256", data.password.encode("utf-8"), bytes(16), PBKDF2_ITERATIONS)
    if account is None or not valid or not account["is_active"]:
        set_hod_login_failure(client)
        raise HTTPException(status_code=401, detail="HOD ID/email or password is incorrect.")
    HOD_LOGIN_FAILURES.pop(client, None)
    HOD_LOGIN_LOCKOUTS.pop(client, None)
    attendance_db.update_hod_last_login(account["hod_id"])
    set_hod_session(account["hod_id"], response, secure=request.url.scheme == "https")
    return {"ok": True, "department": account["department"]}


@app.post("/hod/auth/logout")
def hod_logout(request: Request, response: Response):
    token = request.cookies.get(HOD_SESSION_COOKIE, "")
    with hod_session_lock:
        HOD_SESSIONS.pop(token, None)
    response.delete_cookie(HOD_SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return {"ok": True}


@app.get("/hod-accounts")
def hod_account_list(request: Request):
    ensure_admin(request)
    return attendance_db.hod_account_list()


@app.post("/hod-accounts")
def create_hod_account(data: HODAccountCreate, request: Request):
    ensure_admin(request)
    email = data.email.strip().lower()
    department = data.department.strip()
    full_name = data.full_name.strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid HOD email address.")
    if attendance_db.hod_account_for_login(email) is not None:
        raise HTTPException(status_code=409, detail="A HOD account already uses that email.")
    salt, password_hash = employee_password_hash(data.password)
    try:
        hod_id = attendance_db.create_hod_account(
            full_name, email, department, salt, password_hash
        )
    except (sqlite3.IntegrityError, psycopg.IntegrityError) as exc:
        raise HTTPException(status_code=409, detail="A HOD account already uses that email.") from exc
    return {"ok": True, "hod_id": hod_id}


@app.patch("/hod-accounts/{hod_id}")
def update_hod_account(hod_id: str, data: HODAccountUpdate, request: Request):
    ensure_admin(request)
    if attendance_db.hod_account_by_id(hod_id) is None:
        raise HTTPException(status_code=404, detail="HOD account not found.")
    updates = data.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Provide an account update.")
    if "email" in updates:
        updates["email"] = updates["email"].strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", updates["email"]):
            raise HTTPException(status_code=400, detail="Enter a valid HOD email address.")
    if "department" in updates:
        updates["department"] = updates["department"].strip()
    if "full_name" in updates:
        updates["full_name"] = updates["full_name"].strip()
    if "is_active" in updates:
        updates["is_active"] = int(updates["is_active"])
    if "password" in updates:
        password = updates.pop("password")
        updates["password_salt"], updates["password_hash"] = employee_password_hash(password)
    try:
        if not attendance_db.update_hod_account(hod_id, updates):
            raise HTTPException(status_code=404, detail="HOD account not found.")
    except (sqlite3.IntegrityError, psycopg.IntegrityError) as exc:
        raise HTTPException(status_code=409, detail="A HOD account already uses that email.") from exc
    if "password_hash" in updates or updates.get("is_active") == 0:
        revoke_hod_sessions(hod_id)
    return {"ok": True}


@app.delete("/hod-accounts/{hod_id}")
def delete_hod_account(hod_id: str, request: Request):
    ensure_admin(request)
    if not attendance_db.delete_hod_account(hod_id):
        raise HTTPException(status_code=404, detail="HOD account not found.")
    revoke_hod_sessions(hod_id)
    return {"ok": True}


def _hod_department_employees(hod):
    department = (hod["department"] or "").strip()
    return [
        employee for employee in attendance_db.employees(active_only=False)
        if (employee["department"] or "").strip().casefold() == department.casefold()
    ]


def _ensure_hod_employee(hod, employee_id):
    employee = attendance_db.employee_by_id(employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    if (employee["department"] or "").strip().casefold() != (hod["department"] or "").strip().casefold():
        raise HTTPException(status_code=403, detail="Employee is outside your department.")
    return employee


@app.get("/hod/dashboard")
def hod_dashboard(request: Request):
    hod = ensure_hod(request)
    attendance_db.detect_missing_checkouts(database_path=attendance_db.DATABASE_FILE)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    department_employees = _hod_department_employees(hod)
    employee_ids = {item["employee_id"] for item in department_employees}
    snapshot = attendance_db.daily_attendance(today.isoformat())
    records = [row for row in snapshot["records"] if row["employee_id"] in employee_ids]
    present_statuses = {"ON_TIME", "GRACE_PERIOD", "LATE", "CURRENTLY_WORKING", "CHECKED_OUT"}
    leave_requests = attendance_db.leave_requests()
    approved_on_leave = {
        row["employee_id"] for row in leave_requests
        if row["status"] == "approved"
        and row["start_date"] <= today.isoformat() <= row["end_date"]
        and row["employee_id"] in employee_ids
    }
    return {
        "hod": {"hod_id": hod["hod_id"], "full_name": hod["full_name"], "department": hod["department"]},
        "department": hod["department"],
        "stats": {
            "total_employees": len(department_employees),
            "present_today": sum(row["status"] in present_statuses for row in records),
            "absent_today": sum(row["status"] == "ABSENT" for row in records),
            "late_today": sum(row.get("attendance_status") == "LATE" for row in records),
            "currently_working": sum(row["status"] == "CURRENTLY_WORKING" for row in records),
            "checked_out": sum(row["status"] == "CHECKED_OUT" for row in records),
            "on_leave": len(approved_on_leave),
            "missing_checkouts": attendance_db.missing_checkout_count(hod["department"]),
            "pending_corrections": attendance_db.pending_correction_count(hod["department"]),
        },
        "employees": [
            employee_profile_response(employee) for employee in department_employees
        ],
        "attendance": records,
    }


class OffsiteMeetingCreate(BaseModel):
    attendance_date: str
    departure_time: str
    return_time: str | None = None
    meeting_title: str = Field(min_length=3, max_length=200)
    meeting_details: str = Field(default="", max_length=1000)
    location: str = Field(default="", max_length=200)


class OffsiteMeetingReview(BaseModel):
    status: Literal["approved", "rejected"]
    reviewer_note: str = Field(default="", max_length=500)


@app.get("/daily-attendance")
def daily_attendance_page(request: Request, start_date: str = "", end_date: str = "",
                          department: str = "", employee_id: str = "", status: str = ""):
    ensure_admin(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    start = parse_iso_date(start_date, "Start date must use YYYY-MM-DD.") if start_date else today
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else start
    if end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")
    return attendance_db.day_wise_attendance(start.isoformat(), end.isoformat(), department, employee_id, status)


@app.get("/daily-attendance/export")
def export_daily_attendance(request: Request, start_date: str = "", end_date: str = "",
                            department: str = "", format: Literal["csv", "xlsx", "pdf"] = "csv"):
    ensure_admin(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    start = parse_iso_date(start_date, "Start date must use YYYY-MM-DD.") if start_date else today
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else start
    if end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")
    content, media_type, filename = chatbot_service.build_report(
        {"kind": "attendance", "start_date": start.isoformat(), "end_date": end.isoformat(), "department": department},
        format,
    )
    return Response(content, media_type=media_type,
                    headers={"Cache-Control": "private, no-store", "Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/hod/daily-attendance")
def hod_daily_attendance(request: Request, start_date: str = "", end_date: str = "",
                         employee_id: str = "", status: str = ""):
    hod = ensure_hod(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    start = parse_iso_date(start_date, "Start date must use YYYY-MM-DD.") if start_date else today
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else start
    if end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")
    return attendance_db.day_wise_attendance(start.isoformat(), end.isoformat(), hod["department"], employee_id, status)


@app.get("/hod/daily-attendance/export")
def export_hod_daily_attendance(request: Request, start_date: str = "", end_date: str = ""):
    hod = ensure_hod(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    start = parse_iso_date(start_date, "Start date must use YYYY-MM-DD.") if start_date else today
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else start
    records = attendance_db.day_wise_attendance(start.isoformat(), end.isoformat(), hod["department"])
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Employee ID", "Employee", "Department", "Date", "Check-in", "Checkout", "Working hours", "Break duration", "Off-site duration", "Attendance status", "Punctuality", "Verification"])
    for row in records:
        writer.writerow([row["employee_id"], row["full_name"], row["department"], row["attendance_date"], row["first_check_in"], row["final_checkout"], row["working_hours"], row["break_duration"], row["offsite_duration"], row["attendance_status"], row["punctuality_status"], row["checkout_verification_status"]])
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Cache-Control": "private, no-store", "Content-Disposition": 'attachment; filename="hod-daily-attendance.csv"'})


@app.get("/employee/daily-attendance")
def employee_daily_attendance(request: Request, start_date: str = "", end_date: str = ""):
    employee = ensure_employee(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    start = parse_iso_date(start_date, "Start date must use YYYY-MM-DD.") if start_date else today
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else start
    if end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")
    return attendance_db.day_wise_attendance(start.isoformat(), end.isoformat(), "", employee["employee_id"])


@app.get("/employee/daily-attendance/export")
def export_employee_daily_attendance(request: Request, start_date: str = "", end_date: str = ""):
    employee = ensure_employee(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    start = parse_iso_date(start_date, "Start date must use YYYY-MM-DD.") if start_date else today
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else start
    records = attendance_db.day_wise_attendance(start.isoformat(), end.isoformat(), "", employee["employee_id"])
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Date", "Check-in", "Checkout", "Working hours", "Break duration", "Off-site duration", "Attendance status", "Punctuality", "Verification"])
    for row in records:
        writer.writerow([row["attendance_date"], row["first_check_in"], row["final_checkout"], row["working_hours"], row["break_duration"], row["offsite_duration"], row["attendance_status"], row["punctuality_status"], row["checkout_verification_status"]])
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Cache-Control": "private, no-store", "Content-Disposition": 'attachment; filename="my-daily-attendance.csv"'})


@app.post("/employee/offsite-meetings")
def submit_offsite_meeting(data: OffsiteMeetingCreate, request: Request):
    employee = ensure_employee(request)
    attendance_day = parse_iso_date(data.attendance_date, "Date must use YYYY-MM-DD.")
    try:
        departure = datetime_time.fromisoformat(data.departure_time)
        return_time = datetime_time.fromisoformat(data.return_time) if data.return_time else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Meeting times must use HH:MM.") from exc
    if not re.fullmatch(r"\d{2}:\d{2}", data.departure_time):
        raise HTTPException(status_code=400, detail="Meeting times must use HH:MM.")
    if data.return_time and not re.fullmatch(r"\d{2}:\d{2}", data.return_time):
        raise HTTPException(status_code=400, detail="Meeting times must use HH:MM.")
    config = attendance_db.settings()
    timezone = ZoneInfo(config["timezone"])
    departure_at = datetime.combine(attendance_day, departure, timezone).isoformat(timespec="seconds")
    return_at = None
    if return_time:
        return_day = attendance_day
        if return_time <= departure:
            return_day += timedelta(days=1)
        return_at = datetime.combine(return_day, return_time, timezone).isoformat(timespec="seconds")
    meeting_id = attendance_db.create_offsite_meeting(
        employee["employee_id"], attendance_day.isoformat(), departure_at,
        data.meeting_title.strip(), data.meeting_details.strip(), data.location.strip(), return_at,
    )
    attendance_db.add_notification("hod", "", employee["employee_id"], "offsite_meeting",
                                   "New off-site meeting request",
                                   f"{employee['full_name']} submitted an off-site meeting request for {attendance_day.isoformat()}.")
    attendance_db.add_notification("hr", "", employee["employee_id"], "offsite_meeting",
                                   "New off-site meeting request",
                                   f"{employee['full_name']} submitted an off-site meeting request for {attendance_day.isoformat()}.")
    return {"meeting_id": meeting_id}


@app.get("/hod/offsite-meetings")
def hod_offsite_meetings(request: Request):
    hod = ensure_hod(request)
    return attendance_db.offsite_meeting_records(department=hod["department"])


@app.patch("/hod/offsite-meetings/{meeting_id}")
def review_hod_offsite_meeting(meeting_id: int, data: OffsiteMeetingReview, request: Request):
    hod = ensure_hod(request)
    meetings = attendance_db.offsite_meeting_records(status="pending")
    meeting = next((item for item in meetings if item["meeting_id"] == meeting_id), None)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Off-site meeting request not found.")
    if (meeting["department"] or "").strip().casefold() != (hod["department"] or "").strip().casefold():
        raise HTTPException(status_code=403, detail="Off-site request is outside your department.")
    updated = attendance_db.review_offsite_meeting(
        meeting_id, data.status, data.reviewer_note.strip(), hod["hod_id"], hod["full_name"], "hod",
    )
    if not updated:
        raise HTTPException(status_code=409, detail="Off-site meeting request is not pending.")
    attendance_db.reconcile_daily_attendance_summaries(attendance_db.DATABASE_FILE)
    return {"ok": True}


@app.get("/hr/offsite-meetings")
def hr_offsite_meetings(request: Request):
    ensure_admin(request)
    return attendance_db.offsite_meeting_records()


@app.patch("/hr/offsite-meetings/{meeting_id}")
def review_hr_offsite_meeting(meeting_id: int, data: OffsiteMeetingReview, request: Request):
    ensure_admin(request)
    updated = attendance_db.review_offsite_meeting(
        meeting_id, data.status, data.reviewer_note.strip(), "HR", "HR Administrator", "hr",
    )
    if not updated:
        raise HTTPException(status_code=409, detail="Off-site meeting request is not pending.")
    attendance_db.reconcile_daily_attendance_summaries(attendance_db.DATABASE_FILE)
    return {"ok": True}


@app.get("/hod/employees/{employee_id}")
def hod_employee_detail(employee_id: str, request: Request):
    hod = ensure_hod(request)
    employee = _ensure_hod_employee(hod, employee_id)
    return {
        "profile": employee_profile_response(employee),
        "attendance": employee_attendance_month(employee_id, datetime.now().strftime("%Y-%m")),
        "leave": employee_leave_data(employee_id),
    }


@app.get("/hod/employees/{employee_id}/attendance")
def hod_employee_attendance(employee_id: str, month: str, request: Request):
    hod = ensure_hod(request)
    _ensure_hod_employee(hod, employee_id)
    try:
        date.fromisoformat(month + "-01")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Use YYYY-MM for month.") from exc
    return employee_attendance_month(employee_id, month)


@app.get("/hod/employees/{employee_id}/profile-photo")
def hod_employee_profile_photo(employee_id: str, request: Request):
    hod = ensure_hod(request)
    employee = _ensure_hod_employee(hod, employee_id)
    path = safe_profile_photo_path(employee)
    if path is None:
        raise HTTPException(status_code=404, detail="Profile photo is unavailable.")
    return FileResponse(path, media_type="image/jpeg", content_disposition_type="inline",
                        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@app.get("/hod/attendance/{attendance_id}/photo/{action}")
def hod_attendance_photo(attendance_id: int, action: Literal["check_in", "check_out"], request: Request):
    hod = ensure_hod(request)
    record = attendance_db.attendance_record(attendance_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attendance photo not found.")
    _ensure_hod_employee(hod, record["employee_id"])
    relative_path = attendance_db.attendance_photo_path(attendance_id, action)
    if not relative_path:
        raise HTTPException(status_code=404, detail="No attendance photo is available.")
    try:
        path = safe_attendance_photo_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Attendance photo path is invalid.") from exc
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Attendance photo is no longer available.")
    return FileResponse(path, media_type="image/jpeg", content_disposition_type="inline",
                        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@app.get("/hod/reports/export")
def hod_report_export(request: Request, period: Literal["daily", "weekly", "monthly"] = "monthly",
                     month: str = "", date_value: str = ""):
    hod = ensure_hod(request)
    config = attendance_db.settings()
    timezone = ZoneInfo(config["timezone"])
    today = datetime.now(timezone).date()
    if period == "daily":
        start = date.fromisoformat(date_value) if date_value else today
        end = start
    elif period == "weekly":
        end = date.fromisoformat(date_value) if date_value else today
        start = end - timedelta(days=6)
    else:
        if month:
            start = date.fromisoformat(month + "-01")
        else:
            start = today.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if end > today:
            end = today
    employee_ids = {item["employee_id"] for item in _hod_department_employees(hod)}
    records = attendance_db.attendance_records(start.isoformat(), end.isoformat(), "", hod["department"])
    records = [row for row in records if row["employee_id"] in employee_ids]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Employee ID", "Employee", "Department", "Date", "Status", "Check-in", "Check-out", "Working hours", "Late minutes", "Early departure minutes"])
    for row in records:
        writer.writerow([
            row["employee_id"], row["full_name"], row["department"] or "Unassigned",
            row["attendance_date"], row.get("attendance_status") or row["status"],
            row.get("check_in") or "", row.get("check_out") or "",
            row.get("working_hours") if row.get("working_hours") is not None else "",
            row.get("late_minutes") if row.get("late_minutes") is not None else "",
            row.get("early_departure_minutes") if row.get("early_departure_minutes") is not None else "",
        ])
    filename = f"hod-{hod['department'].replace(' ', '-')}-{period}-{start}-{end}.csv"
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Cache-Control": "private, no-store", "Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/recognize")
def recognize(file: UploadFile = File(...)):
    raw_data = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(raw_data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    data = np.frombuffer(raw_data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read the image.")
    if img.shape[0] * img.shape[1] > 40_000_000:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")
    results = []
    with lock:
        now = time.time()
        for token, proof in list(RECOGNITION_PROOFS.items()):
            if recognition_proof_expired(proof, now):
                RECOGNITION_PROOFS.pop(token, None)
        for name, expires in list(ACCESS_COOLDOWNS.items()):
            if expires <= now:
                ACCESS_COOLDOWNS.pop(name, None)
        for face in detect(img):
            name, score = identify(embed(img, face))
            recognized_at = time.time()
            reason = access_reason(name)
            cooldown_remaining = max(0.0, ACCESS_COOLDOWNS.get(name, 0) - now)
            x, y, w, h = [int(v) for v in face[:4]]
            result = {
                "name": name,
                "score": round(score, 3),
                "allowed": reason is None and cooldown_remaining == 0,
                "reason": reason or "",
                "cooldown": reason is None and cooldown_remaining > 0,
                "cooldown_remaining": cooldown_remaining,
                "liveness_verified": False,
                "liveness_score": None,
                "liveness_status": "photo_check_required",
                "box": [x, y, w, h],
            }
            employee = attendance_db.employee_by_face_name(name) if name != "Unknown" else None
            result["employee_id"] = employee["employee_id"] if employee else ""
            result["attendance_allowed"] = bool(
                employee
                and employee["status"] == "active"
                and employee["consent_at"]
            )
            if reason is None and cooldown_remaining == 0:
                token = secrets.token_urlsafe(24)
                RECOGNITION_PROOFS[token] = (
                    name, round(score, 3), time.time() + 10, recognized_at,
                )
                result["proof"] = token
            elif name != "Unknown" and STAFF_META.get(name, default_meta()).get("active", True):
                token = secrets.token_urlsafe(24)
                RECOGNITION_PROOFS[token] = (
                    name, round(score, 3), time.time() + 10, recognized_at,
                )
                result["proof"] = token
            results.append(result)
    return {"faces": results}


class Access(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)
    proofs: list[str] = Field(min_length=3, max_length=3)


@app.post("/access")
def access(data: Access):
    with lock:
        now = time.time()
        for token, proof in list(RECOGNITION_PROOFS.items()):
            if recognition_proof_expired(proof, now):
                RECOGNITION_PROOFS.pop(token, None)
        evidence = [RECOGNITION_PROOFS.pop(token, None) for token in data.proofs]
        valid_evidence = (
            all(item is not None for item in evidence)
            and len(set(data.proofs)) == 3
            and all(item[0] == data.name and item[2] >= now for item in evidence if item)
            and evidence[-1][1] == data.score
        )
        if not valid_evidence:
            raise HTTPException(status_code=403, detail="Access denied: fresh face recognition is required.")
        reason = access_reason(data.name)
        if reason is None:
            cooldown_remaining = ACCESS_COOLDOWNS.get(data.name, 0) - now
            if cooldown_remaining > 0:
                raise HTTPException(
                    status_code=429,
                    detail=f"Please wait {math.ceil(cooldown_remaining)} seconds before granting access to {data.name} again.",
                )
            ACCESS_COOLDOWNS[data.name] = now + ACCESS_COOLDOWN_SECONDS
    if reason:
        append_entry("entry", "denied", data.name, data.score, "face", reason)
        raise HTTPException(status_code=403, detail=f"Access denied: {reason}.")
    append_entry("entry", "granted", data.name, data.score, "face")
    return {"ok": True}


class Denied(BaseModel):
    name: str = "Unknown"
    score: float = 0
    reason: str = "unknown"


@app.post("/denied")
def denied(data: Denied = Denied()):
    allowed_reasons = {"unknown", "inactive", "outside allowed hours"}
    reason = data.reason if data.reason in allowed_reasons else "unknown"
    append_entry("entry", "denied", data.name[:80], data.score, "face", reason)
    return {"ok": True}


def validate_schedule(start, end):
    pattern = r"(?:[01]\d|2[0-3]):[0-5]\d"
    if not re.fullmatch(pattern, start) or not re.fullmatch(pattern, end):
        raise HTTPException(status_code=400, detail="Use 24-hour HH:MM for both schedule times.")


def parse_iso_date(value, message):
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=message) from exc
    if parsed.isoformat() != value:
        raise HTTPException(status_code=400, detail=message)
    return parsed


def update_employee_record(employee, updates, meta_updates=None):
    if not meta_updates:
        if not attendance_db.update_employee(employee["employee_id"], updates):
            raise HTTPException(status_code=500, detail="Employee profile could not be updated.")
        return

    name = employee["face_name"]
    with lock:
        stored_meta = STAFF_META.get(name)
        previous_meta = dict(stored_meta or default_meta())
        STAFF_META[name] = {**previous_meta, **meta_updates}
        metadata_saved = False
        try:
            save_meta()
            metadata_saved = True
            if not attendance_db.update_employee(employee["employee_id"], updates):
                raise HTTPException(
                    status_code=500, detail="Employee profile could not be updated."
                )
        except Exception:
            if stored_meta is None:
                STAFF_META.pop(name, None)
            else:
                STAFF_META[name] = stored_meta
            if metadata_saved:
                save_meta()
            raise


class StaffSettings(BaseModel):
    name: str
    active: bool
    start: str = "00:00"
    end: str = "23:59"


@app.get("/staff")
def staff_list():
    return {name: len(matrix) for name, matrix in STAFF.items()}


@app.get("/staff/details")
def staff_details(request: Request):
    ensure_admin(request)
    result = []
    for name in sorted(STAFF):
        meta = STAFF_META.get(name, default_meta())
        if _supabase_storage_enabled():
            objects = face_storage.list_objects(
                face_storage.BUCKET_EMPLOYEE_FACES, _staff_storage_prefix(name)
            )
            files = sorted(
                item.get("name", "") for item in objects
                if os.path.splitext(item.get("name", ""))[1].lower() in IMG_EXT + VID_EXT
            )
        else:
            folder = os.path.join(STAFF_DIR, name)
            files = sorted(
                file for file in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, file))
                and os.path.splitext(file)[1].lower() in IMG_EXT + VID_EXT
            )
        result.append({
            "name": name,
            "samples": len(STAFF[name]),
            "files": files,
            "active": meta.get("active", True),
            "start": meta.get("start", "00:00"),
            "end": meta.get("end", "23:59"),
        })
    return result


@app.post("/staff")
def add_staff(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...),
    department: str = Form(""),
    designation: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    joining_date: str = Form(""),
    consent: bool = Form(False),
):
    ensure_admin(request)
    global STAFF
    name = name.strip()
    if not valid_name(name):
        raise HTTPException(
            status_code=400,
            detail="Use a name with letters, numbers, spaces, apostrophes, periods, hyphens, or underscores.",
        )
    if not consent:
        raise HTTPException(status_code=400, detail="Employee consent is required before face enrollment.")
    if len(department) > 100 or len(designation) > 100 or len(phone) > 40 or len(email) > 254:
        raise HTTPException(status_code=400, detail="Employee profile fields exceed their maximum lengths.")
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if joining_date:
        try:
            date.fromisoformat(joining_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Joining date must use YYYY-MM-DD.") from exc
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Choose a photo to upload.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Photo must be 10 MB or smaller.")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="Could not read that photo.") from exc
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read that photo.")
    if img.shape[0] * img.shape[1] > 40_000_000:
        raise HTTPException(status_code=413, detail="Photo dimensions are too large.")

    with lock:
        faces = detect(img)
        if len(faces) != 1:
            raise HTTPException(status_code=400, detail="Photo must contain exactly one detectable face.")
        face = faces[0]
        x, y, width, height = (int(value) for value in face[:4])
        if width < 90 or height < 90:
            raise HTTPException(status_code=400, detail="Face is too small. Move closer and try again (at least 90x90 pixels).")
        left, top = max(x, 0), max(y, 0)
        crop = img[top:min(top + height, img.shape[0]), left:min(left + width, img.shape[1])]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 35 or float(np.mean(gray)) > 220:
            raise HTTPException(status_code=400, detail="Face lighting is too dark or bright. Improve the lighting and try again.")
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 20:
            raise HTTPException(status_code=400, detail="Photo looks blurry. Hold still and try a sharper photo.")
        embedding = embed(img, face)
    ok, encoded = cv2.imencode(".jpg", img)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not process that photo.")
    filename = f"{uuid.uuid4().hex}.jpg"
    if _supabase_storage_enabled():
        image_path = f"{_staff_storage_prefix(name)}/{filename}"
        face_storage.upload_bytes(
            face_storage.BUCKET_EMPLOYEE_FACES, image_path,
            encoded.tobytes(), "image/jpeg",
        )
    else:
        folder = os.path.join(STAFF_DIR, name)
        os.makedirs(folder, exist_ok=True)
        image_path = os.path.join(folder, filename)
        with open(image_path, "xb") as image_file:
            image_file.write(encoded.tobytes())
        if os.name == "posix":
            os.chmod(image_path, 0o600)
    try:
        employee_id = attendance_db.upsert_enrolled_employee(
            name,
            {
                "full_name": name,
                "department": department.strip(),
                "designation": designation.strip(),
                "email": email.strip(),
                "phone": phone.strip(),
                "joining_date": joining_date,
                "profile_photo": image_path,
            },
        )
    except Exception:
        if _supabase_storage_enabled():
            face_storage.delete_objects(face_storage.BUCKET_EMPLOYEE_FACES, [image_path])
        elif os.path.exists(image_path):
            os.remove(image_path)
        raise
    with lock:
        samples = list(STAFF.get(name, []))
        samples.append(embedding)
        STAFF[name] = np.stack(samples)
        STAFF_META.setdefault(name, default_meta())
        save_meta()
        sample_count = len(STAFF[name])
    return {"employee_id": employee_id, "name": name, "samples": sample_count}


@app.patch("/staff")
def update_staff(settings: StaffSettings, request: Request):
    ensure_admin(request)
    if settings.name not in STAFF:
        raise HTTPException(status_code=404, detail="Staff member not found.")
    employee = attendance_db.employee_by_face_name(settings.name)
    if employee is None:
        raise HTTPException(status_code=500, detail="Employee profile is not synchronized.")
    validate_schedule(settings.start, settings.end)
    update_employee_record(
        employee,
        {"status": "active" if settings.active else "inactive"},
        {"active": settings.active, "start": settings.start, "end": settings.end},
    )
    return {"ok": True}


class RenameStaff(BaseModel):
    old_name: str
    new_name: str


@app.post("/staff/rename")
def rename_staff(data: RenameStaff, request: Request):
    ensure_admin(request)
    global STAFF
    old_name, new_name = data.old_name.strip(), data.new_name.strip()
    if old_name not in STAFF:
        raise HTTPException(status_code=404, detail="Staff member not found.")
    if not valid_name(new_name):
        raise HTTPException(status_code=400, detail="Please enter a valid staff name.")
    if new_name != old_name and (
        new_name in STAFF or (
            not _supabase_storage_enabled()
            and os.path.exists(os.path.join(STAFF_DIR, new_name))
        )
    ):
        raise HTTPException(status_code=409, detail="A staff member with that name already exists.")
    if old_name != new_name:
        if _supabase_storage_enabled():
            old_prefix = _staff_storage_prefix(old_name)
            new_prefix = _staff_storage_prefix(new_name)
            objects = face_storage.list_objects(
                face_storage.BUCKET_EMPLOYEE_FACES, old_prefix
            )
            moved = []
            try:
                for item in objects:
                    filename = item.get("name", "")
                    if not filename:
                        continue
                    source = f"{old_prefix}/{filename}"
                    destination = f"{new_prefix}/{filename}"
                    face_storage.move_object(
                        face_storage.BUCKET_EMPLOYEE_FACES, source, destination
                    )
                    moved.append((source, destination))
                attendance_db.set_employee_face_name(old_name, new_name)
            except Exception:
                for source, destination in reversed(moved):
                    try:
                        face_storage.move_object(
                            face_storage.BUCKET_EMPLOYEE_FACES, destination, source
                        )
                    except Exception:
                        pass
                raise
        else:
            old_folder = os.path.join(STAFF_DIR, old_name)
            new_folder = os.path.join(STAFF_DIR, new_name)
            if os.path.islink(old_folder) or os.path.commonpath([os.path.realpath(STAFF_DIR), os.path.realpath(old_folder)]) != os.path.realpath(STAFF_DIR):
                raise HTTPException(status_code=400, detail="Staff folder is not a safe path.")
            os.rename(old_folder, new_folder)
            try:
                attendance_db.set_employee_face_name(old_name, new_name)
            except Exception:
                os.rename(new_folder, old_folder)
                raise
        with lock:
            STAFF[new_name] = STAFF.pop(old_name)
            STAFF_META[new_name] = STAFF_META.pop(old_name, default_meta())
            save_meta()
    return {"ok": True, "name": new_name}


class RemovePhoto(BaseModel):
    name: str
    file: str


@app.post("/staff/photo/delete")
def delete_staff_photo(data: RemovePhoto, request: Request):
    ensure_admin(request)
    global STAFF
    if data.name not in STAFF or data.file != os.path.basename(data.file):
        raise HTTPException(status_code=404, detail="Photo not found.")
    if os.path.splitext(data.file)[1].lower() not in IMG_EXT + VID_EXT:
        raise HTTPException(status_code=400, detail="Unsupported sample type.")
    if _supabase_storage_enabled():
        object_path = f"{_staff_storage_prefix(data.name)}/{data.file}"
        objects = face_storage.list_objects(
            face_storage.BUCKET_EMPLOYEE_FACES, _staff_storage_prefix(data.name)
        )
        if not any(item.get("name") == data.file for item in objects):
            raise HTTPException(status_code=404, detail="Photo not found.")
        face_storage.delete_objects(face_storage.BUCKET_EMPLOYEE_FACES, [object_path])
    else:
        folder = os.path.join(STAFF_DIR, data.name)
        path = os.path.join(folder, data.file)
        if not os.path.isfile(path) or os.path.islink(path):
            raise HTTPException(status_code=404, detail="Photo not found.")
        os.remove(path)
    with lock:
        loaded = load_staff()
        STAFF = loaded
        if data.name not in STAFF:
            STAFF_META.pop(data.name, None)
            save_meta()
            attendance_db.erase_employee_biometrics(data.name)
    return {"ok": True}


class DeleteStaff(BaseModel):
    name: str


@app.post("/staff/delete")
def delete_staff(data: DeleteStaff, request: Request):
    ensure_admin(request)
    global STAFF
    name = data.name
    if name not in STAFF:
        raise HTTPException(status_code=404, detail="Staff member not found.")
    if _supabase_storage_enabled():
        prefix = _staff_storage_prefix(name)
        objects = face_storage.list_objects(face_storage.BUCKET_EMPLOYEE_FACES, prefix)
        paths = [f"{prefix}/{item['name']}" for item in objects if item.get("name")]
        face_storage.delete_objects(face_storage.BUCKET_EMPLOYEE_FACES, paths)
    else:
        folder = os.path.join(STAFF_DIR, name)
        staff_root = os.path.realpath(STAFF_DIR)
        if os.path.islink(folder) or os.path.commonpath([staff_root, os.path.realpath(folder)]) != staff_root:
            raise HTTPException(status_code=400, detail="Staff folder is not a safe path.")
        shutil.rmtree(folder)
    with lock:
        STAFF.pop(name, None)
        STAFF_META[name] = {**STAFF_META.get(name, default_meta()), "active": False}
        save_meta()
    attendance_db.erase_employee_biometrics(name)
    return {"ok": True}


@app.post("/reload")
def reload_staff(request: Request):
    ensure_admin(request)
    global STAFF
    with lock:
        STAFF = load_staff()
        for name in STAFF:
            STAFF_META.setdefault(name, default_meta())
        save_meta()
    return {"staff": sorted(STAFF.keys())}


class EmployeeProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=120)
    department: str | None = Field(default=None, max_length=100)
    designation: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=254)
    phone: str | None = Field(default=None, max_length=40)
    joining_date: str | None = None
    shift_id: int | None = Field(default=None, ge=1)
    status: Literal["active", "inactive"] | None = None
    consent: bool | None = None


@app.get("/employees")
def employee_list(request: Request):
    ensure_admin(request)
    employees = attendance_db.employees()
    samples = {name: len(matrix) for name, matrix in STAFF.items()}
    return [
        {
            **{key: value for key, value in employee.items()
               if key not in {"profile_photo", "consent_at"}},
            "consent_recorded": bool(employee["consent_at"]),
            "samples": samples.get(employee["face_name"], 0),
            "profile_photo_url": (
                f"/employees/{employee['employee_id']}/profile-photo"
                if employee["profile_photo"] else None
            ),
        }
        for employee in employees
    ]


def safe_profile_photo_path(employee):
    photo = employee.get("profile_photo")
    if not photo:
        return None
    staff_root = os.path.realpath(STAFF_DIR)
    path = os.path.realpath(photo if os.path.isabs(photo) else os.path.join(".", photo))
    try:
        safe = os.path.commonpath([staff_root, path]) == staff_root
    except ValueError:
        safe = False
    if not safe or not os.path.isfile(path):
        return None
    return path


@app.get("/employees/{employee_id}/profile-photo")
def hr_employee_profile_photo(employee_id: str, request: Request):
    ensure_admin(request)
    employee = attendance_db.employee_by_id(employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    if _supabase_storage_enabled():
        object_path = employee.get("profile_photo") or ""
        try:
            content = face_storage.download_bytes(
                face_storage.BUCKET_EMPLOYEE_FACES, object_path
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Profile photo is unavailable.") from exc
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
    path = safe_profile_photo_path(employee)
    if path is None:
        raise HTTPException(status_code=404, detail="Profile photo is unavailable.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/employee-accounts")
def employee_account_list(request: Request):
    ensure_admin(request)
    return attendance_db.employee_account_list()


@app.post("/employee-accounts")
def create_employee_account(
    data: EmployeeAccountCreate, request: Request
):
    ensure_admin(request)
    if attendance_db.employee_by_id(data.employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    salt, password_hash = employee_password_hash(data.password)
    try:
        created = attendance_db.create_employee_account(
            data.employee_id, salt, password_hash, data.annual_leave_days
        )
    except (sqlite3.IntegrityError, psycopg.IntegrityError) as exc:
        raise HTTPException(
            status_code=409, detail="An employee login account already exists."
        ) from exc
    if not created:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return {"ok": True}


@app.patch("/employee-accounts/{employee_id}")
def update_employee_account(
    employee_id: str, data: EmployeeAccountUpdate, request: Request
):
    ensure_admin(request)
    if attendance_db.employee_by_id(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    updates = data.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Provide an account update.")
    if "is_active" in updates:
        if updates["is_active"] is None:
            raise HTTPException(status_code=400, detail="Account status cannot be null.")
        updates["is_active"] = int(updates["is_active"])
    if "password" in updates:
        password = updates.pop("password")
        if password is None:
            raise HTTPException(status_code=400, detail="Password cannot be null.")
        salt, password_hash = employee_password_hash(password)
        updates["password_salt"] = salt
        updates["password_hash"] = password_hash
    if not attendance_db.update_employee_account(employee_id, updates):
        raise HTTPException(status_code=404, detail="Employee login account not found.")
    if "password_hash" in updates or updates.get("is_active") == 0:
        revoke_employee_sessions(employee_id)
    return {"ok": True}


@app.patch("/employees/{employee_id}")
def update_employee_profile(
    employee_id: str, data: EmployeeProfileUpdate, request: Request
):
    ensure_admin(request)
    employee = attendance_db.employee_by_id(employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    updates = data.model_dump(exclude_unset=True)
    for key, value in updates.items():
        if value is None:
            raise HTTPException(status_code=400, detail=f"{key} cannot be null.")
    for key in ("full_name", "department", "designation", "email", "phone"):
        if key in updates:
            updates[key] = updates[key].strip()
    if "full_name" in updates and not updates["full_name"]:
        raise HTTPException(status_code=400, detail="Full name cannot be empty.")
    if "consent" in updates:
        consent = updates.pop("consent")
        if consent is False:
            raise HTTPException(
                status_code=400,
                detail="Use biometric data deletion to withdraw consent and remove face data.",
            )
        if consent is True:
            updates["consent_at"] = datetime.now(ZoneInfo("Asia/Qatar")).isoformat(
                timespec="seconds"
            )
    if "joining_date" in updates and updates["joining_date"]:
        parse_iso_date(updates["joining_date"], "Joining date must use YYYY-MM-DD.")
    if updates.get("email") and not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+", updates["email"]
    ):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if "shift_id" in updates:
        shift = next(
            (
                item for item in attendance_db.list_shifts()
                if item["shift_id"] == updates["shift_id"]
            ),
            None,
        )
        if shift is None:
            raise HTTPException(status_code=404, detail="Shift not found.")
        if not shift["active"] and updates["shift_id"] != employee["shift_id"]:
            raise HTTPException(
                status_code=400,
                detail="Employees can only be assigned to active shifts.",
            )
    if updates:
        meta_updates = (
            {"active": updates["status"] == "active"}
            if "status" in updates else None
        )
        update_employee_record(employee, updates, meta_updates)
    if "consent_at" in updates:
        append_entry(
            "biometric_consent",
            "recorded",
            employee["face_name"],
            "",
            "admin",
            "Employee consent recorded",
        )
    return {"ok": True}


@app.delete("/employees/{employee_id}/biometrics")
def erase_employee_biometrics(employee_id: str, request: Request):
    ensure_admin(request)
    employee = attendance_db.employee_by_id(employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    name = employee["face_name"]
    folder = os.path.join(STAFF_DIR, name)
    if os.path.isdir(folder):
        staff_root = os.path.realpath(STAFF_DIR)
        if os.path.islink(folder) or os.path.commonpath(
            [staff_root, os.path.realpath(folder)]
        ) != staff_root:
            raise HTTPException(status_code=400, detail="Staff folder is not a safe path.")
        shutil.rmtree(folder)
    with lock:
        STAFF.pop(name, None)
        STAFF_META[name] = {**STAFF_META.get(name, default_meta()), "active": False}
        save_meta()
    attendance_db.erase_employee_biometrics(name)
    append_entry("biometric_data", "deleted", name, "", "admin", "Consent withdrawn")
    return {"ok": True}


class AttendanceClock(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    action: Literal["check_in", "check_out"]
    proofs: list[str] = Field(min_length=3, max_length=3)
    photo_data: str | None = Field(
        default=None, min_length=1, max_length=MAX_ATTENDANCE_PHOTO_DATA_LENGTH
    )


class ManualAttendance(BaseModel):
    employee_id: str = Field(min_length=1, max_length=32)
    action: Literal["check_in", "check_out"]
    reason: str = Field(min_length=3, max_length=200)


def attendance_message(employee, record, action):
    timestamp = record.get("attendance_action_time") or (
        record["check_in"] if action == "check_in" else record["check_out"]
    )
    local_time = datetime.fromisoformat(timestamp).strftime("%I:%M %p")
    status = attendance_db.classify_attendance(record)["attendance_status"]
    photo_key = "checkin_photo_path" if action == "check_in" else "checkout_photo_path"
    photo_notice = (
        f" {action.replace('_', '-').title()} photo captured successfully."
        if record.get(photo_key) else ""
    )
    liveness_key = (
        "checkin_liveness_score" if action == "check_in"
        else "checkout_liveness_score"
    )
    liveness_score = record.get(liveness_key)
    if liveness_score is not None:
        photo_notice += (
            f" Liveness verification passed (score {liveness_score:.2f})."
        )
    if action == "check_in" and record.get("returned_from_break"):
        first_checkin = datetime.fromisoformat(record["check_in"]).strftime("%I:%M %p")
        return (
            f"Welcome back {employee['full_name']} at {local_time}! Your first "
            f"check-in remains {first_checkin}. The earlier exit was treated as a "
            "temporary break."
            f"{photo_notice}",
            "granted",
        )
    if action == "check_in" and status == "LATE":
        late_minutes = attendance_db.classify_attendance(record)["late_minutes"]
        return (
            f"⚠️ Late: {employee['full_name']} checked in at {local_time}, "
            f"{late_minutes} minute(s) after the late cutoff. "
            f"Attendance was recorded.{photo_notice}",
            "late",
        )
    if action == "check_in" and status == "GRACE_PERIOD":
        return (
            f"⚠️ Warning: {employee['full_name']} checked in during the grace period "
            f"at {local_time}. Attendance was recorded. "
            f"The on-time window is {record['on_time_window_minutes']} minutes and "
            f"the late cutoff is {record['grace_period_minutes']} minutes after shift start."
            f"{photo_notice}",
            "warning",
        )
    if action == "check_in":
        return (
            f"Welcome {employee['full_name']}! Check-in successful at {local_time}."
            f"{photo_notice}",
            "granted",
        )
    early_minutes = attendance_db.classify_attendance(record)[
        "early_departure_minutes"
    ]
    if early_minutes:
        return (
            f"⚠ Early departure: {employee['full_name']} checked out at {local_time}, "
            f"{early_minutes} minute(s) before the scheduled shift end. "
            f"If this was a temporary break, check in again on return."
            f"{photo_notice}",
            "warning",
        )
    return (
        f"Goodbye {employee['full_name']}! Check-out successful at {local_time}."
        f"{photo_notice}",
        "granted",
    )


def attendance_spoken_message(employee, action):
    action_text = "checked in" if action == "check_in" else "checked out"
    return f"{employee['full_name']} {action_text}"


@app.post("/attendance/clock")
def attendance_clock(data: AttendanceClock):
    if not data.photo_data:
        raise HTTPException(
            status_code=400,
            detail="Live attendance photo is missing. Refresh the page and try again.",
        )
    with lock:
        now = time.time()
        for token, proof in list(RECOGNITION_PROOFS.items()):
            if recognition_proof_expired(proof, now):
                RECOGNITION_PROOFS.pop(token, None)
        evidence = [RECOGNITION_PROOFS.pop(token, None) for token in data.proofs]
        valid_evidence = (
            all(item is not None for item in evidence)
            and len(set(data.proofs)) == 3
            and all(item[0] == data.name and item[2] >= now for item in evidence if item)
        )
        if not valid_evidence:
            raise HTTPException(status_code=403, detail="Fresh face recognition is required.")
        if not evidence[0][3] < evidence[1][3] < evidence[2][3]:
            raise HTTPException(
                status_code=403,
                detail="Face recognition must be confirmed across fresh camera frames.",
            )
        score = evidence[-1][1]
        recognized_at = evidence[-1][3]

    employee = attendance_db.employee_by_face_name(data.name)
    if employee is None or employee["status"] != "active":
        raise HTTPException(status_code=403, detail="Employee is not active.")
    if not employee["consent_at"]:
        raise HTTPException(
            status_code=403,
            detail="Recorded employee consent is required before face attendance.",
        )
    photo_bytes, photo_score, liveness_score = decode_attendance_photo(
        data.photo_data, data.name
    )
    timezone_name = attendance_db.settings()["timezone"]
    captured_at = datetime.fromtimestamp(recognized_at, ZoneInfo(timezone_name))
    if data.action == "check_out":
        checkin_photo_path = attendance_db.open_attendance_photo_path(
            employee["employee_id"]
        )
        if checkin_photo_path:
            checkin_path = safe_attendance_photo_path(checkin_photo_path)
            if os.path.isfile(checkin_path):
                with open(checkin_path, "rb") as file:
                    checkin_bytes = file.read(MAX_ATTENDANCE_PHOTO_BYTES + 1)
                if hashlib.sha256(checkin_bytes).digest() == hashlib.sha256(photo_bytes).digest():
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Check-out photo must be a new live camera frame; "
                            "attendance was not recorded."
                        ),
                    )
    try:
        photo_path = save_attendance_photo(
            data.action, employee["employee_id"], captured_at, photo_bytes
        )
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="Attendance photo could not be saved; attendance was not recorded.",
        ) from exc

    try:
        record = attendance_db.record_attendance(
            employee["employee_id"],
            data.action,
            now=captured_at,
            timezone_name=timezone_name,
            photo_path=photo_path,
            recognition_score=photo_score,
            liveness_score=liveness_score,
        )
    except FileExistsError as exc:
        try:
            remove_attendance_photo_file(photo_path)
        except OSError as cleanup_error:
            print(f"Could not remove unreferenced attendance photo: {cleanup_error}")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        try:
            remove_attendance_photo_file(photo_path)
        except OSError as cleanup_error:
            print(f"Could not remove unreferenced attendance photo: {cleanup_error}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        try:
            remove_attendance_photo_file(photo_path)
        except OSError as cleanup_error:
            print(f"Could not remove unreferenced attendance photo: {cleanup_error}")
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        try:
            remove_attendance_photo_file(photo_path)
        except OSError as cleanup_error:
            print(f"Could not remove unreferenced attendance photo: {cleanup_error}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (sqlite3.Error, psycopg.Error) as exc:
        try:
            remove_attendance_photo_file(photo_path)
        except OSError as cleanup_error:
            print(f"Could not remove unreferenced attendance photo: {cleanup_error}")
        raise HTTPException(
            status_code=500,
            detail="Attendance could not be saved; its photo was not retained.",
        ) from exc

    status = "checked_in" if data.action == "check_in" else "checked_out"
    append_entry("attendance", status, data.name, score, "face")
    message, tone = attendance_message(employee, record, data.action)
    say(attendance_spoken_message(employee, data.action))
    return {
        "ok": True,
        "message": message,
        "tone": tone,
        "attendance": attendance_record_response(record),
    }


@app.post("/attendance/manual")
def manual_attendance(data: ManualAttendance, request: Request):
    ensure_admin(request)
    reason = data.reason.strip()
    if not reason:
        raise HTTPException(status_code=400, detail="A reason is required for the audit log.")
    employee = attendance_db.employee_by_id(data.employee_id)
    if employee is None or employee["status"] != "active":
        raise HTTPException(status_code=404, detail="Select an active employee.")
    try:
        record = attendance_db.record_attendance(
            employee["employee_id"],
            data.action,
            timezone_name=attendance_db.settings()["timezone"],
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_entry(
        "attendance",
        "checked_in" if data.action == "check_in" else "checked_out",
        employee["face_name"],
        "",
        "manual",
        reason,
    )
    message, tone = attendance_message(employee, record, data.action)
    say(attendance_spoken_message(employee, data.action))
    return {
        "ok": True,
        "message": message,
        "tone": tone,
        "attendance": attendance_record_response(record),
    }


@app.get("/dashboard/stats")
def dashboard_summary():
    attendance_db.detect_missing_checkouts(database_path=attendance_db.DATABASE_FILE)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date().isoformat()
    result = attendance_db.dashboard_stats(today)
    result["missing_checkouts"] = attendance_db.missing_checkout_count()
    result["pending_corrections"] = attendance_db.pending_correction_count()
    return result


def attendance_record_response(record):
    result = dict(record)
    for path_key, available_key in (
        ("checkin_photo_path", "checkin_photo_available"),
        ("checkout_photo_path", "checkout_photo_available"),
    ):
        relative_path = result.pop(path_key, None)
        if relative_path:
            path = safe_attendance_photo_path(relative_path)
            result[available_key] = os.path.isfile(path)
        else:
            result[available_key] = False
    return result


def employee_attendance_month(employee_id, month):
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=400, detail="Month must use YYYY-MM.")
    try:
        month_start = date.fromisoformat(f"{month}-01")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Month must use YYYY-MM.") from exc
    month_end = month_start.replace(
        day=calendar.monthrange(month_start.year, month_start.month)[1]
    )
    legacy = attendance_db.legacy_attendance_records(
        month_start.isoformat(),
        month_end.isoformat(),
        employee_id=employee_id,
    )
    legacy_by_date = {}
    for record in legacy:
        legacy_by_date.setdefault(record["attendance_date"], []).append(record)

    records = []
    current_day = month_start
    while current_day <= month_end:
        daily = attendance_db.daily_attendance(
            current_day.isoformat(), employee_id=employee_id
        )
        item = next(
            (
                row for row in daily["records"]
                if row["employee_id"] == employee_id
            ),
            None,
        )
        day_legacy = legacy_by_date.get(current_day.isoformat(), [])
        if item is None:
            item = {
                "employee_id": employee_id,
                "attendance_date": current_day.isoformat(),
                "status": "NOT_SCHEDULED",
                "check_in": None,
                "check_out": None,
            }
        elif not item.get("attendance_id") and not item.get("check_in") and day_legacy:
            check_in = min(day_legacy, key=lambda row: row["check_in"])["check_in"]
            check_outs = [row["check_out"] for row in day_legacy if row["check_out"]]
            worked_hours = sum(
                row["working_hours"] or 0 for row in day_legacy
            )
            item = {
                **item,
                "attendance_date": current_day.isoformat(),
                "check_in": check_in,
                "check_out": max(check_outs) if check_outs else None,
                "working_hours": round(worked_hours, 2) if worked_hours else None,
                "status": "CHECKED_OUT" if check_outs else "CURRENTLY_WORKING",
                "legacy_event": True,
            }
        item.setdefault("attendance_date", current_day.isoformat())
        if item.get("attendance_id"):
            item = attendance_record_response(item)
        else:
            item = {
                **item,
                "checkin_photo_available": False,
                "checkout_photo_available": False,
            }
        records.append(item)
        current_day += timedelta(days=1)

    present_statuses = {
        "ON_TIME", "GRACE_PERIOD", "LATE", "CURRENTLY_WORKING", "CHECKED_OUT"
    }
    summary = {
        "present_days": sum(row["status"] in present_statuses for row in records),
        "absent_days": sum(row["status"] == "ABSENT" for row in records),
        "late_days": sum(row.get("attendance_status") == "LATE" for row in records),
        "working_hours": round(
            sum(row.get("working_hours") or 0 for row in records), 2
        ),
    }
    return {"month": month, "summary": summary, "records": records}


def employee_profile_response(employee):
    return {
        "employee_id": employee["employee_id"],
        "full_name": employee["full_name"],
        "department": employee["department"],
        "designation": employee["designation"],
        "email": employee["email"],
        "phone": employee["phone"],
        "joining_date": employee["joining_date"],
        "profile_photo_url": (
            "/employee/profile-photo" if employee.get("profile_photo") else None
        ),
    }


def employee_leave_data(employee_id):
    config = attendance_db.settings()
    year = datetime.now(ZoneInfo(config["timezone"])).year
    return {
        "balance": attendance_db.leave_balance(employee_id, year),
        "requests": attendance_db.leave_requests(employee_id=employee_id),
    }


def employee_notifications(leave_requests, corrections):
    notifications = []
    for item in leave_requests:
        notifications.append({
            "type": "leave",
            "title": f"Leave request {item['status']}",
            "message": f"{item['start_date']} to {item['end_date']}",
            "created_at": item["reviewed_at"] or item["created_at"],
        })
    for item in corrections:
        notifications.append({
            "type": "attendance",
            "title": f"Attendance correction {item['status']}",
            "message": item["attendance_date"],
            "created_at": item["reviewed_at"] or item["created_at"],
        })
    return sorted(
        notifications, key=lambda item: item["created_at"], reverse=True
    )[:20]


@app.get("/employee/profile")
def employee_profile(request: Request):
    employee = ensure_employee(request)
    return employee_profile_response(employee)


@app.get("/employee/profile-photo")
def employee_profile_photo(request: Request):
    employee = ensure_employee(request)
    path = safe_profile_photo_path(employee)
    if path is None:
        raise HTTPException(status_code=404, detail="Profile photo is unavailable.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/employee/attendance")
def employee_attendance(request: Request, month: str = ""):
    employee = ensure_employee(request)
    if not month:
        config = attendance_db.settings()
        month = datetime.now(ZoneInfo(config["timezone"])).strftime("%Y-%m")
    return employee_attendance_month(employee["employee_id"], month)


@app.get("/employee/attendance/export")
def export_employee_attendance(request: Request, month: str = ""):
    employee = ensure_employee(request)
    if not month:
        config = attendance_db.settings()
        month = datetime.now(ZoneInfo(config["timezone"])).strftime("%Y-%m")
    data = employee_attendance_month(employee["employee_id"], month)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "Date", "Status", "Check-in", "Check-out", "Working hours", "Late minutes"
    ])
    for item in data["records"]:
        writer.writerow([
            item["attendance_date"],
            item.get("attendance_status") or item["status"],
            item.get("check_in") or "",
            item.get("check_out") or "",
            item.get("working_hours") if item.get("working_hours") is not None else "",
            item.get("late_minutes") if item.get("late_minutes") is not None else "",
        ])
    filename = f"attendance-{employee['employee_id']}-{month}.csv"
    return Response(
        output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/employee/attendance/{attendance_id}/photo/{action}")
def employee_attendance_photo(
    attendance_id: int,
    action: Literal["check_in", "check_out"],
    request: Request,
):
    employee = ensure_employee(request)
    record = attendance_db.attendance_record(attendance_id)
    if record is None or record["employee_id"] != employee["employee_id"]:
        raise HTTPException(status_code=404, detail="Attendance photo not found.")
    relative_path = attendance_db.attendance_photo_path(attendance_id, action)
    if not relative_path:
        raise HTTPException(status_code=404, detail="No attendance photo is available.")
    try:
        path = safe_attendance_photo_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Attendance photo path is invalid.") from exc
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Attendance photo is no longer available.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/employee/attendance/{attendance_id}/movement/{movement_id}/photo")
def employee_attendance_movement_photo(
    attendance_id: int,
    movement_id: int,
    request: Request,
):
    employee = ensure_employee(request)
    record = attendance_db.attendance_record(attendance_id)
    if record is None or record["employee_id"] != employee["employee_id"]:
        raise HTTPException(status_code=404, detail="Attendance photo not found.")
    relative_path = attendance_db.attendance_movement_photo_path(
        attendance_id, movement_id
    )
    if not relative_path:
        raise HTTPException(status_code=404, detail="No movement photo is available.")
    try:
        path = safe_attendance_photo_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Movement photo path is invalid.") from exc
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Movement photo is no longer available.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/employee/leave-requests")
def employee_leave_requests(request: Request):
    employee = ensure_employee(request)
    return employee_leave_data(employee["employee_id"])


class EmployeeLeaveRequestCreate(BaseModel):
    start_date: str
    end_date: str
    reason: str = Field(min_length=3, max_length=500)


@app.post("/employee/leave-requests")
def submit_employee_leave_request(
    data: EmployeeLeaveRequestCreate, request: Request
):
    employee = ensure_employee(request)
    start = parse_iso_date(data.start_date, "Dates must use YYYY-MM-DD.")
    end = parse_iso_date(data.end_date, "Dates must use YYYY-MM-DD.")
    if end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")
    reason = data.reason.strip()
    if len(reason) < 3:
        raise HTTPException(status_code=400, detail="Reason must contain at least 3 non-space characters.")
    leave_id = attendance_db.create_leave_request(
        employee["employee_id"], start.isoformat(), end.isoformat(), reason
    )
    return {"leave_id": leave_id}


class EmployeeCorrectionRequestCreate(BaseModel):
    attendance_date: str
    requested_check_in: str | None = None
    requested_check_out: str | None = None
    request_type: Literal[
        "Forgot to checkout",
        "Official company meeting",
        "Official off-site work",
        "Other",
    ] = "Forgot to checkout"
    reason: str = Field(min_length=3, max_length=500)
    supporting_document_name: str | None = Field(default=None, max_length=255)
    supporting_document_base64: str | None = Field(default=None, max_length=MAX_ATTENDANCE_PHOTO_DATA_LENGTH)


@app.get("/employee/correction-requests")
def employee_correction_requests(request: Request):
    employee = ensure_employee(request)
    return attendance_db.attendance_correction_requests(
        employee_id=employee["employee_id"]
    )


@app.post("/employee/correction-requests")
def submit_employee_correction_request(
    data: EmployeeCorrectionRequestCreate, request: Request
):
    employee = ensure_employee(request)
    attendance_date = parse_iso_date(data.attendance_date, "Date must use YYYY-MM-DD.")
    if not data.requested_check_in and not data.requested_check_out:
        raise HTTPException(
            status_code=400, detail="Provide a corrected check-in or check-out time."
        )
    for time_value in (data.requested_check_in, data.requested_check_out):
        if time_value:
            if not re.fullmatch(r"\d{2}:\d{2}", time_value):
                raise HTTPException(
                    status_code=400, detail="Correction times must use HH:MM."
                )
            try:
                datetime_time.fromisoformat(time_value)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Correction times must use HH:MM."
                ) from exc
    reason = data.reason.strip()
    if len(reason) < 3:
        raise HTTPException(status_code=400, detail="Reason must contain at least 3 non-space characters.")
    config = attendance_db.settings()
    timezone = ZoneInfo(config["timezone"])
    today = datetime.now(timezone).date()
    if attendance_date > today:
        raise HTTPException(status_code=400, detail="Attendance correction dates cannot be in the future.")
    period_days = int(config["attendance_correction_period_days"])
    if (today - attendance_date).days > period_days:
        raise HTTPException(
            status_code=400,
            detail=f"Corrections are allowed only within {period_days} days.",
        )
    records = attendance_db.attendance_records(
        data.attendance_date, data.attendance_date,
        employee_id=employee["employee_id"],
    )
    if not records:
        raise HTTPException(status_code=404, detail="No attendance record exists for that date.")
    record = records[0]
    if data.requested_check_out:
        check_in = datetime.fromisoformat(record["check_in"])
        check_out_time = datetime_time.fromisoformat(data.requested_check_out)
        check_out_day = attendance_date
        if check_out_time <= check_in.timetz().replace(tzinfo=None):
            check_out_day += timedelta(days=1)
        check_out = datetime.combine(check_out_day, check_out_time, timezone)
        if check_out <= check_in:
            raise HTTPException(status_code=400, detail="Checkout time must be later than check-in.")
    document_path = ""
    if data.supporting_document_base64:
        if not data.supporting_document_name:
            raise HTTPException(status_code=400, detail="Supporting document name is required.")
        try:
            document_bytes = base64.b64decode(data.supporting_document_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Supporting document is invalid base64.") from exc
        if len(document_bytes) > MAX_ATTENDANCE_PHOTO_BYTES:
            raise HTTPException(status_code=413, detail="Supporting document is too large.")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", data.supporting_document_name)[-255:]
        os.makedirs(CORRECTION_DOCUMENT_DIR, mode=0o700, exist_ok=True)
        document_path = os.path.join(
            CORRECTION_DOCUMENT_DIR,
            f"{uuid.uuid4().hex}-{safe_name}",
        )
        with open(document_path, "xb") as document_file:
            document_file.write(document_bytes)
        if os.name == "posix":
            os.chmod(document_path, 0o600)
    correction_id = attendance_db.create_attendance_correction_request(
        employee["employee_id"], data.attendance_date,
        data.requested_check_in, data.requested_check_out,
        reason, data.request_type, document_path,
    )
    if correction_id is None:
        if document_path:
            try:
                os.remove(document_path)
            except OSError:
                pass
        raise HTTPException(status_code=409, detail="A pending or approved correction already exists for this attendance date.")
    employee_name = employee["full_name"]
    attendance_db.add_notification(
        "employee", employee["employee_id"], employee["employee_id"],
        "attendance_correction", "Correction request submitted",
        f"Your correction request for {data.attendance_date} was submitted for review.",
    )
    attendance_db.add_notification(
        "hod", "", employee["employee_id"], "attendance_correction",
        "New attendance correction request",
        f"{employee_name} submitted a correction request for {data.attendance_date}.",
    )
    attendance_db.add_notification(
        "hr", "", employee["employee_id"], "attendance_correction",
        "New attendance correction request",
        f"{employee_name} submitted a correction request for {data.attendance_date}.",
    )
    return {"correction_id": correction_id}


@app.post("/employee/change-password")
def change_employee_password(
    data: EmployeePasswordChange, request: Request, response: Response
):
    employee = ensure_employee(request)
    account = attendance_db.employee_account(employee["employee_id"])
    if account is None or not employee_password_matches(
        data.current_password, account["password_salt"], account["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if data.current_password == data.new_password:
        raise HTTPException(
            status_code=400, detail="Choose a new password different from the current password."
        )
    salt, password_hash = employee_password_hash(data.new_password)
    if not attendance_db.update_employee_account(
        employee["employee_id"],
        {"password_salt": salt, "password_hash": password_hash},
    ):
        raise HTTPException(status_code=500, detail="Password could not be changed.")
    revoke_employee_sessions(employee["employee_id"])
    clear_employee_session(request, response)
    return {"ok": True, "reauthenticate": True}


@app.get("/employee/dashboard")
def employee_dashboard(request: Request, month: str = ""):
    employee = ensure_employee(request)
    attendance_db.detect_missing_checkouts(database_path=attendance_db.DATABASE_FILE)
    if not month:
        config = attendance_db.settings()
        month = datetime.now(ZoneInfo(config["timezone"])).strftime("%Y-%m")
    attendance = employee_attendance_month(employee["employee_id"], month)
    leave_data = employee_leave_data(employee["employee_id"])
    corrections = attendance_db.attendance_correction_requests(
        employee_id=employee["employee_id"]
    )
    return {
        "profile": employee_profile_response(employee),
        "attendance": attendance,
        "leave_balance": leave_data["balance"],
        "leave_requests": leave_data["requests"],
        "correction_requests": corrections,
        "notifications": (
            employee_notifications(leave_data["requests"], corrections)
            + [
                {
                    "type": "attendance",
                    "title": item["title"],
                    "message": item["message"],
                    "created_at": item["created_at"],
                }
                for item in attendance_db.notifications("employee", employee["employee_id"])
            ]
        )
    }


@app.get("/attendance")
def attendance_history(
    request: Request,
    start_date: str = "",
    end_date: str = "",
    name: str = "",
    department: str = "",
    employee_id: str = "",
    status: str = "",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    sort: Literal["check_in", "late_minutes", "name"] = "check_in",
    direction: Literal["asc", "desc"] = "desc",
):
    ensure_admin(request)
    validate_filters(start_date, end_date)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    selected_day = date.fromisoformat(end_date or start_date) if end_date or start_date else today
    if start_date and end_date and start_date != end_date:
        first = date.fromisoformat(start_date)
        if (selected_day - first).days > 365:
            raise HTTPException(status_code=400, detail="Attendance history is limited to one year.")
    all_records, records, summary, counts, punctuality_counts, total_scheduled, percentages = (
        attendance_monitor_data(
            start_date, end_date, selected_day, name, department, employee_id, status
        )
    )
    sort_field = {
        "check_in": lambda row: row.get("check_in") or "",
        "late_minutes": lambda row: row.get("late_minutes", 0),
        "name": lambda row: row.get("full_name", "").casefold(),
    }[sort]
    records.sort(key=sort_field, reverse=direction == "desc")
    total = len(records)
    start = (page - 1) * page_size
    return {
        "date": selected_day.isoformat(),
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": math.ceil(total / page_size),
        "records": [
            attendance_record_response(record)
            for record in records[start:start + page_size]
        ],
        "counts": counts,
        "punctuality_counts": punctuality_counts,
        "total_scheduled": total_scheduled,
        "percentages": percentages,
    }


def attendance_monitor_data(
    start_date, end_date, selected_day, name="", department="", employee_id="", status=""
):
    filters = {
        "name": name.casefold().strip(),
        "employee_id": employee_id.casefold().strip(),
        "department": department.casefold().strip(),
    }

    def matches_identity(record):
        return (
            (not filters["name"] or filters["name"] in record["full_name"].casefold())
            and (not filters["employee_id"] or filters["employee_id"] in record["employee_id"].casefold())
            and (not filters["department"] or filters["department"] in (record["department"] or "").casefold())
        )

    daily = attendance_db.daily_attendance(selected_day)
    summary_legacy = attendance_db.legacy_attendance_records(
        selected_day.isoformat(), selected_day.isoformat(), name
    )
    summary_legacy_days = {
        (row["employee_id"], row["attendance_date"]) for row in summary_legacy
    }
    summary = [
        row for row in daily["records"]
        if matches_identity(row)
        and (
            row.get("attendance_id")
            or row.get("check_in")
            or (
                row["employee_id"],
                row.get("attendance_date", selected_day.isoformat()),
            ) not in summary_legacy_days
        )
    ]
    counts = {category: 0 for category in attendance_db.ATTENDANCE_STATUSES}
    punctuality_counts = {
        category: 0 for category in ("ON_TIME", "GRACE_PERIOD", "LATE", "ABSENT")
    }
    for row in summary:
        counts[row["status"]] += 1
        if row.get("attendance_status") in punctuality_counts:
            punctuality_counts[row["attendance_status"]] += 1
        elif row["status"] == "ABSENT":
            punctuality_counts["ABSENT"] += 1
    scheduled_statuses = {
        "ON_TIME", "GRACE_PERIOD", "LATE", "CURRENTLY_WORKING",
        "CHECKED_OUT", "ABSENT", "NOT_CHECKED_IN",
    }
    total_scheduled = sum(counts[category] for category in scheduled_statuses)
    punctuality_population = sum(punctuality_counts.values())
    percentages = {
        category: round(100 * punctuality_counts[category] / punctuality_population, 1)
        if punctuality_population else 0
        for category in ("ON_TIME", "GRACE_PERIOD", "LATE", "ABSENT")
    }
    is_single_day = (
        bool(start_date or end_date)
        and (not start_date or not end_date or start_date == end_date)
    )
    if is_single_day:
        all_records = daily["records"]
    else:
        all_records = attendance_db.attendance_records(
            start_date, end_date, name, department
        )
        for record in all_records:
            record["record_status"] = record["status"]
            record["status"] = record["attendance_status"]

    legacy_records = attendance_db.legacy_attendance_records(start_date, end_date, name)
    recorded_employee_days = {
        (row["employee_id"], row["attendance_date"])
        for row in all_records
        if row.get("attendance_id") and row.get("check_in")
    }
    legacy_employee_days = {
        (row["employee_id"], row["attendance_date"])
        for row in legacy_records
    }
    all_records = [
        row for row in all_records
        if row.get("attendance_id")
        or row.get("check_in")
        or (
            row["employee_id"],
            row.get("attendance_date", selected_day.isoformat()),
        ) not in legacy_employee_days
    ]
    all_records.extend(
        row for row in legacy_records
        if (row["employee_id"], row["attendance_date"]) not in recorded_employee_days
    )
    all_records = [row for row in all_records if matches_identity(row)]
    allowed_statuses = attendance_db.ATTENDANCE_STATUSES | {"ALL"}
    if status and status not in allowed_statuses:
        raise HTTPException(status_code=400, detail="Unknown attendance status filter.")
    records = all_records
    if status and status != "ALL":
        if status in {"ON_TIME", "GRACE_PERIOD", "LATE"}:
            records = [
                row for row in records
                if row.get("attendance_status") == status
            ]
        elif not is_single_day and status in {
            "ABSENT", "NOT_CHECKED_IN", "ON_LEAVE", "WEEKLY_OFF", "HOLIDAY",
            "NOT_SCHEDULED", "CURRENTLY_WORKING",
        }:
            records = [row for row in summary if row["status"] == status]
        else:
            records = [row for row in records if row["status"] == status]
    return (
        all_records, records, summary, counts, punctuality_counts,
        total_scheduled, percentages,
    )


@app.get("/attendance/record/{attendance_id}")
def attendance_record_detail(attendance_id: int, request: Request):
    ensure_admin(request)
    record = attendance_db.attendance_record(attendance_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attendance record not found.")
    month = date.fromisoformat(record["attendance_date"]).replace(day=1)
    month_end = (
        month.replace(day=28) + timedelta(days=4)
    ).replace(day=1) - timedelta(days=1)
    history = attendance_db.attendance_records(
        month.isoformat(), month_end.isoformat(), name=record["full_name"]
    )
    history = [
        item for item in history if item["employee_id"] == record["employee_id"]
    ]
    movements = []
    for movement in attendance_db.attendance_movements(attendance_id):
        relative_path = movement.pop("photo_path")
        movement["photo_available"] = False
        if relative_path:
            path = safe_attendance_photo_path(relative_path)
            movement["photo_available"] = os.path.isfile(path)
        movements.append(movement)
    return {
        "record": attendance_record_response(record),
        "movements": movements,
        "month_history": [
            attendance_record_response(item) for item in history
        ],
    }


@app.get("/attendance/record/{attendance_id}/movement/{movement_id}/photo")
def attendance_movement_photo(
    attendance_id: int,
    movement_id: int,
    request: Request,
    download: bool = False,
):
    ensure_admin(request)
    relative_path = attendance_db.attendance_movement_photo_path(
        attendance_id, movement_id
    )
    if not relative_path:
        raise HTTPException(status_code=404, detail="No movement photo is available.")
    try:
        path = safe_attendance_photo_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Movement photo path is invalid.") from exc
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Movement photo is no longer available.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=os.path.basename(path),
        content_disposition_type="attachment" if download else "inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/attendance/record/{attendance_id}/photo/{action}")
def attendance_photo(
    attendance_id: int,
    action: Literal["check_in", "check_out"],
    request: Request,
    download: bool = False,
):
    ensure_admin(request)
    relative_path = attendance_db.attendance_photo_path(attendance_id, action)
    if not relative_path:
        raise HTTPException(status_code=404, detail="No attendance photo is available.")
    try:
        path = safe_attendance_photo_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Attendance photo path is invalid.") from exc
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Attendance photo is no longer available.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=os.path.basename(path),
        content_disposition_type="attachment" if download else "inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.delete("/attendance/record/{attendance_id}/photo/{action}")
def delete_attendance_photo(
    attendance_id: int,
    action: Literal["check_in", "check_out"],
    request: Request,
):
    ensure_admin(request)
    record = attendance_db.attendance_record(attendance_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attendance record not found.")
    relative_path = attendance_db.attendance_photo_path(attendance_id, action)
    if not relative_path:
        raise HTTPException(status_code=404, detail="No attendance photo is available.")
    try:
        remove_attendance_photo_file(relative_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Could not delete attendance photo.") from exc
    cleared_path = attendance_db.clear_attendance_photo(attendance_id, action)
    if cleared_path is None:
        raise HTTPException(status_code=404, detail="Attendance record not found.")
    append_entry(
        "attendance_photo",
        "deleted",
        record["face_name"],
        "",
        "admin",
        f"{action.replace('_', '-')} evidence deleted for attendance {attendance_id}",
    )
    return {"ok": True}


@app.delete("/attendance")
def clear_attendance_history(request: Request):
    ensure_admin(request)
    photo_paths = attendance_db.attendance_photo_paths()
    deleted = attendance_db.clear_attendance_records()
    cleanup_errors = 0
    for relative_path in photo_paths:
        try:
            remove_attendance_photo_file(relative_path)
        except (OSError, ValueError) as exc:
            cleanup_errors += 1
            print(f"Could not remove attendance photo during history clear: {exc}")
    return {"ok": True, "deleted": deleted, "photo_cleanup_errors": cleanup_errors}


@app.get("/attendance/export")
def export_attendance(
    request: Request,
    start_date: str = "",
    end_date: str = "",
    name: str = "",
    department: str = "",
    employee_id: str = "",
    status: str = "",
):
    ensure_admin(request)
    validate_filters(start_date, end_date)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    selected_day = date.fromisoformat(end_date or start_date) if end_date or start_date else today
    if start_date and end_date and start_date != end_date:
        first = date.fromisoformat(start_date)
        if (selected_day - first).days > 365:
            raise HTTPException(status_code=400, detail="Attendance history is limited to one year.")
    _, all_records, _, _, _, _, _ = attendance_monitor_data(
        start_date, end_date, selected_day, name, department, employee_id, status
    )
    fields = [
        "employee_id", "full_name", "department", "attendance_date",
        "shift_start", "check_in", "check_out", "working_hours",
        "grace_period_minutes", "grace_minutes_used", "late_minutes",
        "status", "early_departure", "early_departure_minutes",
        "checkin_photo_available",
        "checkout_photo_available", "checkin_liveness_score",
        "checkout_liveness_score", "remarks",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in all_records:
        punctuality = (
            attendance_db.classify_attendance(record)
            if record.get("check_in") and not record.get("legacy_event") else {}
        )
        status_value = record.get("status") or record.get("attendance_status", "")
        remarks = {
            "ON_TIME": "Checked in on or before shift start.",
            "GRACE_PERIOD": "Checked in within the configured grace period.",
            "LATE": "Checked in after the configured grace period.",
            "ABSENT": "No check-in after the attendance deadline.",
            "CURRENTLY_WORKING": "Checked in; no check-out recorded yet.",
            "ON_LEAVE": "Approved leave.",
            "WEEKLY_OFF": "Not scheduled to work.",
            "HOLIDAY": "Public holiday.",
            "NOT_CHECKED_IN": "Attendance deadline has not passed.",
            "CHECKED_OUT": "Check-out recorded.",
        }.get(status_value, "")
        if record.get("early_departure_minutes", 0):
            remarks = (
                f"EARLY DEPARTURE: {record['early_departure_minutes']} minute(s) "
                "before scheduled shift end."
            )
        photo_metadata = attendance_record_response(record)
        row_values = {
            **record,
            **punctuality,
            "checkin_photo_available": photo_metadata["checkin_photo_available"],
            "checkout_photo_available": photo_metadata["checkout_photo_available"],
            "status": status_value.replace("_", " "),
            "remarks": remarks,
        }
        for key in fields:
            value = str(row_values.get(key, "") or "")
            if value.startswith(("=", "+", "-", "@", "\t", "\r")):
                row_values[key] = "'" + value
        writer.writerow({key: row_values.get(key, "") for key in fields})
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="attendance-report.csv"'},
    )


class ChatbotQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def chatbot_session_key(request):
    token = request.cookies.get(SESSION_COOKIE, "")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@app.get("/api/chatbot/suggestions")
def chatbot_suggestions(request: Request):
    ensure_admin(request)
    return {"suggestions": chatbot_service.SUGGESTIONS}


@app.get("/api/chatbot/history")
def chatbot_history(request: Request):
    ensure_admin(request)
    return {"messages": attendance_db.chatbot_history(chatbot_session_key(request))}


@app.delete("/api/chatbot/history")
def clear_chatbot_history(request: Request):
    ensure_admin(request)
    attendance_db.clear_chatbot_history(chatbot_session_key(request))
    return {"ok": True}


@app.post("/api/chatbot/message")
def chatbot_message(data: ChatbotQuestion, request: Request):
    ensure_admin(request)
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question.")
    session_key = chatbot_session_key(request)
    context = attendance_db.chatbot_context(session_key)
    result = chatbot_service.answer_question(question, context)
    attendance_db.save_chatbot_turn(
        session_key, question, result["answer"], result["context"],
        photos=result.get("photos", []),
    )
    return {
        "answer": result["answer"],
        "report": result["report"],
        "status": result["status"],
        "photos": result.get("photos", []),
        "history": attendance_db.chatbot_history(session_key),
    }


@app.get("/api/chatbot/report")
def chatbot_report(
    request: Request,
    kind: Literal["attendance", "absent", "late"],
    start_date: str,
    end_date: str,
    format: Literal["csv", "xlsx", "pdf"] = "csv",
    department: str = "",
):
    ensure_admin(request)
    validate_filters(start_date, end_date)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if (end - start).days > 365:
        raise HTTPException(status_code=400, detail="Reports are limited to one year.")
    if kind == "absent" and start != end:
        raise HTTPException(
            status_code=400,
            detail="Absent reports require one date because workday and holiday calendars are not configured.",
        )
    try:
        content, media_type, filename = chatbot_service.build_report(
            {
                "kind": kind,
                "start_date": start_date,
                "end_date": end_date,
                "department": department,
            },
            format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/reports/analytics")
def attendance_analytics(
    request: Request,
    start_date: str = "",
    end_date: str = "",
):
    ensure_admin(request)
    config = attendance_db.settings()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    end = parse_iso_date(end_date, "End date must use YYYY-MM-DD.") if end_date else today
    start = (
        parse_iso_date(start_date, "Start date must use YYYY-MM-DD.")
        if start_date else end - timedelta(days=29)
    )
    if start > end:
        raise HTTPException(status_code=400, detail="Start date must be on or before end date.")
    return attendance_db.analytics(start.isoformat(), end.isoformat())


@app.get("/system/settings")
def system_settings(request: Request):
    ensure_admin(request)
    return {"settings": attendance_db.settings(), "shifts": attendance_db.list_shifts()}


class AttendanceSettings(BaseModel):
    office_start: str
    office_end: str
    grace_period_minutes: int = Field(ge=0, le=240)
    standard_work_hours: float = Field(gt=0, le=24)
    timezone: str = "Asia/Qatar"
    attendance_correction_period_days: int = Field(default=30, ge=1, le=365)
    face_threshold: float = Field(ge=0.1, le=0.9)
    working_days: list[str] | None = None
    holidays: list[str] | None = None


@app.patch("/system/settings")
def update_system_settings(data: AttendanceSettings, request: Request):
    ensure_admin(request)
    validate_schedule(data.office_start, data.office_end)
    try:
        ZoneInfo(data.timezone)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Timezone must be a valid IANA timezone.") from exc
    if data.working_days is not None and (
        not data.working_days
        or len(set(data.working_days)) != len(data.working_days)
        or any(day not in attendance_db.WEEKDAYS for day in data.working_days)
    ):
        raise HTTPException(status_code=400, detail="Select at least one valid working day.")
    if data.holidays is not None:
        try:
            holidays = sorted({date.fromisoformat(item).isoformat() for item in data.holidays})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Holiday dates must use YYYY-MM-DD.") from exc
    else:
        holidays = None
    attendance_db.update_default_shift(
        data.office_start,
        data.office_end,
        data.grace_period_minutes,
        data.standard_work_hours,
    )
    settings_update = {
        "timezone": data.timezone,
        "face_threshold": data.face_threshold,
        "attendance_correction_period_days": str(data.attendance_correction_period_days),
    }
    if data.working_days is not None:
        settings_update["working_days"] = json.dumps(data.working_days)
    if holidays is not None:
        settings_update["holidays"] = json.dumps(holidays)
    attendance_db.update_settings(settings_update)
    global THRESHOLD
    THRESHOLD = data.face_threshold
    return {"ok": True}


class ShiftCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    start_time: str
    end_time: str
    grace_period_minutes: int = Field(default=30, ge=0, le=240)
    on_time_window_minutes: int = Field(default=15, ge=0, le=240)
    standard_work_hours: float = Field(default=8, gt=0, le=24)


class ShiftStatusUpdate(BaseModel):
    active: bool


class ShiftTimingUpdate(BaseModel):
    start_time: str
    end_time: str
    grace_period_minutes: int = Field(ge=0, le=240)
    on_time_window_minutes: int | None = Field(default=None, ge=0, le=240)


@app.get("/shifts")
def shift_list(request: Request):
    ensure_admin(request)
    return attendance_db.list_shifts()


@app.post("/shifts")
def add_shift(data: ShiftCreate, request: Request):
    ensure_admin(request)
    name = data.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Shift name cannot be empty.")
    validate_schedule(data.start_time, data.end_time)
    if data.on_time_window_minutes > data.grace_period_minutes:
        raise HTTPException(
            status_code=400,
            detail="On-time window cannot exceed the late-arrival cutoff.",
        )
    try:
        shift_id = attendance_db.create_shift(
            name,
            data.start_time,
            data.end_time,
            data.grace_period_minutes,
            data.standard_work_hours,
            on_time_window_minutes=data.on_time_window_minutes,
        )
    except (sqlite3.IntegrityError, psycopg.IntegrityError) as exc:
        raise HTTPException(status_code=409, detail="A shift with that name already exists.") from exc
    return {"shift_id": shift_id}


@app.put("/shifts/{shift_id}/timings")
def update_shift_times(shift_id: int, data: ShiftTimingUpdate, request: Request):
    ensure_admin(request)
    validate_schedule(data.start_time, data.end_time)
    shift = next(
        (item for item in attendance_db.list_shifts() if item["shift_id"] == shift_id),
        None,
    )
    if shift is None:
        raise HTTPException(status_code=404, detail="Shift not found.")
    on_time_window_minutes = (
        data.on_time_window_minutes
        if data.on_time_window_minutes is not None
        else shift["on_time_window_minutes"]
    )
    if on_time_window_minutes > data.grace_period_minutes:
        raise HTTPException(
            status_code=400,
            detail="On-time window cannot exceed the late-arrival cutoff.",
        )
    updated = attendance_db.update_shift_configuration(
        shift_id, data.start_time, data.end_time, data.grace_period_minutes,
        on_time_window_minutes=on_time_window_minutes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Shift not found.")
    return {
        "ok": True,
        "start_time": data.start_time,
        "end_time": data.end_time,
        "grace_period_minutes": data.grace_period_minutes,
        "on_time_window_minutes": on_time_window_minutes,
    }


@app.delete("/shifts/{shift_id}")
def delete_shift(shift_id: int, request: Request):
    ensure_admin(request)
    try:
        result = attendance_db.remove_shift(shift_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Shift not found.")
    return {"ok": True, **result}


@app.patch("/shifts/{shift_id}")
def update_shift_status(shift_id: int, data: ShiftStatusUpdate, request: Request):
    ensure_admin(request)
    try:
        updated = attendance_db.set_shift_active(shift_id, data.active)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Shift not found.")
    return {"ok": True, "active": data.active}


class LeaveRequestCreate(BaseModel):
    employee_id: str = Field(min_length=1, max_length=32)
    start_date: str
    end_date: str
    reason: str = Field(min_length=3, max_length=500)


class LeaveReview(BaseModel):
    status: Literal["approved", "rejected"]


class AttendanceCorrectionReview(BaseModel):
    status: Literal["approved", "rejected"]
    reviewer_note: str = Field(default="", max_length=500)


@app.get("/hod/attendance-correction-requests")
def hod_attendance_correction_requests(request: Request):
    hod = ensure_hod(request)
    attendance_db.detect_missing_checkouts(database_path=attendance_db.DATABASE_FILE)
    return attendance_db.attendance_correction_requests(department=hod["department"])


@app.patch("/hod/attendance-correction-requests/{correction_id}")
def review_hod_attendance_correction_request(
    correction_id: int, data: AttendanceCorrectionReview, request: Request
):
    hod = ensure_hod(request)
    corrections = attendance_db.attendance_correction_requests(status="pending")
    correction = next((item for item in corrections if item["correction_id"] == correction_id), None)
    if correction is None:
        raise HTTPException(status_code=404, detail="Correction request not found.")
    if (correction["department"] or "").strip().casefold() != (hod["department"] or "").strip().casefold():
        raise HTTPException(status_code=403, detail="Correction request is outside your department.")
    try:
        updated = attendance_db.review_attendance_correction(
            correction_id, data.status, data.reviewer_note.strip(),
            approver_id=hod["hod_id"], approver_name=hod["full_name"],
            approver_role="hod",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=409, detail="Attendance correction request is not pending.")
    attendance_db.reconcile_daily_attendance_summaries(attendance_db.DATABASE_FILE)
    return {"ok": True}


@app.get("/hr/attendance-correction-history")
def hr_attendance_correction_history(request: Request):
    ensure_admin(request)
    return {
        "requests": attendance_db.attendance_correction_requests(),
        "audit": {
            str(item["correction_id"]): attendance_db.correction_audit(item["correction_id"])
            for item in attendance_db.attendance_correction_requests()
        },
    }


@app.get("/leave-requests")
def list_leave_requests(request: Request):
    ensure_admin(request)
    return attendance_db.leave_requests()


@app.post("/leave-requests")
def submit_leave_request(data: LeaveRequestCreate, request: Request):
    ensure_admin(request)
    if attendance_db.employee_by_id(data.employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    start = parse_iso_date(data.start_date, "Dates must use YYYY-MM-DD.")
    end = parse_iso_date(data.end_date, "Dates must use YYYY-MM-DD.")
    if end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")
    reason = data.reason.strip()
    if len(reason) < 3:
        raise HTTPException(status_code=400, detail="Reason must contain at least 3 non-space characters.")
    leave_id = attendance_db.create_leave_request(
        data.employee_id, start.isoformat(), end.isoformat(), reason
    )
    return {"leave_id": leave_id}


@app.patch("/leave-requests/{leave_id}")
def review_leave_request(leave_id: int, data: LeaveReview, request: Request):
    ensure_admin(request)
    if not attendance_db.update_leave_status(leave_id, data.status):
        raise HTTPException(status_code=409, detail="Leave request is not pending.")
    return {"ok": True}



@app.get("/camera/status")
def camera_status():
    return {
        "recognition_model": "OpenCV YuNet + SFace",
        "recognition_ready": True,
        "face_threshold": THRESHOLD,
        "liveness_model": "Passive MiniFASNet ONNX photo check",
        "liveness_ready": liveness_model is not None,
        "liveness_threshold": LIVENESS_THRESHOLD,
    }


@app.get("/entries")
def entry_summary(
    request: Request,
    name: str = "",
    start_date: str = "",
    end_date: str = "",
    status: str = "",
    event: str = "",
):
    validate_filters(start_date, end_date)
    if any((name, start_date, end_date, status, event)):
        ensure_admin(request)
    if not _is_admin_authenticated(request):
        entries = query_entries()
        today_count = sum(
            1 for entry in entries
            if entry["status"] == "granted"
            and entry["event"] == "entry"
            and datetime.fromisoformat(entry["time"]).date() == date.today()
        )
        recent = list(reversed(entries[-10:]))
        return {"today": today_count, "total": len(recent), "entries": recent}
    entries = query_entries(name, start_date, end_date, status, event)
    today = date.today()
    today_count = sum(
        1 for entry in entries
        if entry["status"] == "granted"
        and entry["event"] == "entry"
        and datetime.fromisoformat(entry["time"]).date() == today
    )
    return {"today": today_count, "total": len(entries), "entries": list(reversed(entries[-500:]))}


@app.get("/entries/export")
def export_entries(
    request: Request,
    name: str = "",
    start_date: str = "",
    end_date: str = "",
    status: str = "",
    event: str = "",
):
    ensure_admin(request)
    validate_filters(start_date, end_date)
    entries = query_entries(name, start_date, end_date, status, event)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
    writer.writeheader()
    for entry in entries:
        safe_entry = dict(entry)
        for key in ("event", "status", "name", "method", "details"):
            value = str(safe_entry.get(key, ""))
            if value.startswith(("=", "+", "-", "@", "\t", "\r")):
                safe_entry[key] = "'" + value
        writer.writerow(safe_entry)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="face-access-entries.csv"'},
    )


class ManualAccess(BaseModel):
    name: str
    reason: str = Field(min_length=3, max_length=200)


@app.post("/manual-access")
def manual_access(data: ManualAccess, request: Request):
    ensure_admin(request)
    name = data.name.strip()
    reason = data.reason.strip()
    if name not in STAFF:
        raise HTTPException(status_code=404, detail="Select a registered staff member.")
    if not reason:
        raise HTTPException(status_code=400, detail="A reason is required for the audit log.")
    if not STAFF_META.get(name, default_meta()).get("active", True):
        raise HTTPException(status_code=403, detail="Inactive staff cannot be granted entry.")
    append_entry("entry", "granted", name, "", "manual", reason)
    return {"ok": True}


class VisitorCheckin(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    host: str = Field(min_length=1, max_length=80)
    purpose: str = Field(min_length=1, max_length=200)
    hours: int = Field(ge=1, le=24)


@app.get("/visitors")
def active_visitors(request: Request):
    ensure_admin(request)
    now = datetime.now()
    visitors = []
    for entry in reversed(query_entries(event="visitor_checkin")):
        try:
            expires = datetime.fromisoformat(entry["details"].split(" | Expires: ", 1)[-1])
        except ValueError:
            continue
        if expires > now:
            visitors.append({**entry, "expires": expires.isoformat(timespec="minutes")})
    return visitors


@app.post("/visitors")
def visitor_checkin(data: VisitorCheckin, request: Request):
    ensure_admin(request)
    name, host, purpose = data.name.strip(), data.host.strip(), data.purpose.strip()
    if not name or not host or not purpose:
        raise HTTPException(status_code=400, detail="Visitor, host, and purpose are required.")
    expires = datetime.now() + timedelta(hours=data.hours)
    details = f"Host: {host} | Purpose: {purpose} | Expires: {expires.isoformat(timespec='minutes')}"
    append_entry("visitor_checkin", "checked_in", name, "", "admin", details)
    return {"ok": True, "expires": expires.isoformat(timespec="minutes")}
