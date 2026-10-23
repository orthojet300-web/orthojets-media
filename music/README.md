# Orthojets original music (owned by the channel)
Solo Steinway piano themes composed for Orthojets (tools/piano_theme.py) with the public-domain Splendid Grand Piano
samples (AKAI). No third-party rights -> safe on TikTok, Instagram and YouTube (no Content ID).
Kept in this repo on purpose: the scheduled runs can't always reach the private orthojets-music repo, and episodes must
never go out without music (EP03 and EP04 did).

2026-10-23: the owner cancelled the 3 piano themes (piano_1_rise, piano_2_clarity, piano_3_momentum).

## vault/ (2026-10-23)
Encrypted copies (AES-256, openssl -pbkdf2) of the TikTok/Instagram tracks from the private repo, so scheduled runs that
cannot reach the private repo still have music. Not usable without the key (kept only in the private scheduled-task prompt;
never commit it). build_episode.py decrypts them when ORTHO_MUSIC_KEY is set and ../orthojets-music is missing.
YouTube never uses these (Content ID) — YouTube gets tracks from youtube_safe only (YouTube Audio Library, pending from owner).

vault/yt__*.enc = YouTube Audio Library tracks (YouTube copies only), decrypted into <music>/youtube_safe/.
