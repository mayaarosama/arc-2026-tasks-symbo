"""Where the conductor's web app serves each folder. app.py mounts them; core.py and the task handlers
build output and input URLs from them, so the two can never disagree."""
STATIC = "/static"                 # arcdemo/conductor/static
MEDIA_SESSION = "/media"           # session_dir: live outputs, uploads
MEDIA_REPLAY = "/replay-media"     # replay_dir: rehearsal recordings
MEDIA_INPUT = "/media-input/data"  # data/: stage inputs (pictures, prompts)
LAYOUT_ASSETS = "/layout-assets"   # the furniture catalogue's assets_dir (Task 8 3D models)
