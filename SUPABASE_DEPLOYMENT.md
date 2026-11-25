# Supabase employee face-storage migration

The application uses the private `employee-faces` bucket when both
`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are configured. The service
role key must exist only in trusted server-side environment variables.

## Existing local staff samples

1. Keep a backup of the local `staff/` directory.
2. Configure these variables in your local terminal (do not commit them):
   - `DATABASE_URL`: Supabase PostgreSQL connection string. For a serverless
     deployment, prefer the Supabase transaction pooler connection string.
   - `SUPABASE_URL`: Supabase project URL.
   - `SUPABASE_SERVICE_ROLE_KEY`: Supabase service-role secret.
3. From the repository root, preview the files:
   ```bash
   python migrate_staff_storage.py
   ```
4. If the preview is correct, upload:
   ```bash
   python migrate_staff_storage.py --apply
   ```

The script is safe to re-run for the same file names because it uploads with
upsert enabled. It uploads samples under `employee-faces/<employee-name>/`.
It also updates a matching employee's profile-photo object path when a row
already exists in Supabase. If no employee row exists yet, the uploaded sample
will still be available for the server's staff synchronization step.

## Vercel environment variables

Add the same three variables in the Vercel project's Environment Variables
for the deployment environments you use. Never put the service-role key in
frontend JavaScript, source control, or a `NEXT_PUBLIC_*` variable.

## Storage behavior

- New staff enrollment uploads the processed face image to private storage.
- Face samples are downloaded by the backend when it builds recognition
  embeddings.
- Staff rename, sample deletion, and staff deletion operate on private objects.
- Employee profile photos are streamed through an admin-authenticated API route.
- Without the Supabase storage credentials, local development continues to
  read and write the local `staff/` directory.

This step migrates employee face samples only. Attendance evidence, correction
documents, JSON settings, CSV logs, and serverless session persistence still
need separate migration work before Vercel deployment.
