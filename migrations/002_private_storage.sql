-- Face Access: private Supabase Storage buckets
-- Apply in Supabase SQL Editor. Existing database tables and records are untouched.
-- Access is intended only through the trusted backend using the Supabase service-role key.
-- Never expose the service-role key to browser/client code.

BEGIN;

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES
    ('employee-faces', 'employee-faces', FALSE, 10485760,
     ARRAY['image/jpeg', 'image/png', 'image/webp']),
    ('attendance-evidence', 'attendance-evidence', FALSE, 5242880,
     ARRAY['image/jpeg', 'image/png', 'image/webp']),
    ('correction-documents', 'correction-documents', FALSE, 10485760,
     ARRAY['application/pdf', 'image/jpeg', 'image/png', 'image/webp'])
ON CONFLICT (id) DO UPDATE
SET
    public = FALSE,
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

COMMIT;
