# Orthojets — automated episode pipeline

This repo lets a fresh Claude session make and publish one Orthojets TikTok episode with no human input.
A scheduled task fires **Saturday to Wednesday evenings** (18:58). Each run publishes the next episode at **10:00 the following day** (Africa/Tripoli time), so posts land **Sunday to Thursday at 10:00 — 5 per week** (owner decision 2026-10-19; Friday/Saturday off, the Libyan weekend). The weekly Instagram carousel goes out Sunday 18:00.

## Channel rules (from the owner — do not change)
- All on-screen medical content is in **English medical language**. The intro and outro keep their Arabic tagline.
- **Reading time**: after each text block appears, hold it for 1.5–3.5 s, using the `Timeline` holds in `kit/ep_common.py`.
- **Audio**: a calm licensed background track plays quietly under the episode body (-26 LUFS, fade in and out). The intro and outro keep their own sound.
  - Tracks live in the PRIVATE repo `orthojet300-web/orthojets-music`. Clone it next to this repo as `../orthojets-music`.
  - `tools/build_episode.py` picks the track by episode number: track = ((N-1) mod count) + 1.
  - Only the `*.mp3` files at the TOP LEVEL of the music repo are in rotation (see its README). `blocked/` holds tracks that YouTube
    Content ID claimed (EP02 was blocked worldwide on 2026-10-20) — never use them; `untested/` is not used either.
    Never add a track to the top level unless it has already been on YouTube with no copyright claim (YouTube Audio Library tracks are safe).
  - Never copy the MP3s into this public repo: the license forbids redistributing them on their own.
  - Our own piano themes live in THIS repo under `music/` and are always used (YouTube gets only these). The private repo adds
    the Pixabay tracks to the TikTok/Instagram rotation when it can be reached; if it can't, the build still has music.
  - The music repo is PRIVATE: attach it with `add_repo` access **"push"** (that attaches credentials; "read" does not) and clone it.
    If that fails, retry, then try `gh repo clone`. Music is REQUIRED (owner complaint on EP03, which went out silent):
    use `--no-music` only if every route failed, and say so at the top of the report.
  - After the build, check the body really has music: ebur128 integrated loudness between 10 s and 40 s should be about -30 to -24 LUFS
    (silent bodies measure below -60).
- Brand:
  - Colors: navy `#223F61`, sky `#78AFD8` and white on dark navy. Use green (120,210,170) and coral (236,112,104) only for right-vs-wrong.
  - Fonts: Montserrat ExtraBold/Black. Small text must be at least 27 px.
  - The fonts lack Δ, ≤, ≈ and →. Use `ep_common.rich()` with DejaVuB for Δ, and write the others out in words.
- **Structure (retention-first, from EP03):** `[hook] → [1 s logo flash] → [rest of the episode] → [3 s outro]`.
  - No logo intro at the start. `tools/build_episode.py` assembles this with `kit/add_brand.py --hook-first`.
  - **Hook:** the big headline must be readable within **0.5 s** of the first frame and fully built by about 1.5 s. Pass `hook_end` and `resume` to `main_cli(TL, SCENES, hook_end=..., resume=...)`, both in script time:
    - `hook_end` (usually 2.6–3.0 s): the soft logo dissolves in there.
    - `resume`: the moment the hook has fully faded out and the next scene is starting. The video continues from there after the logo, so the hook never reappears.
    - Background music runs continuously under the logo.
  - Scenes in this order: series-pill hook (see Series), context or diagram, rule plus evidence, Tips & Tricks (3 cards), takeaway with the full reference on screen.
  - Body **57–61 s** with holds, so the finished video is **61–65 s**. That stays above 60 s for TikTok Creator Rewards eligibility while keeping completion high.
  - Change scene or animation every 3–5 s.
  - Keep content within x 60–960 and y 200–1700.
- Anatomy and every number must be accurate. The audience is orthopedic surgeons and medical students.

## Series (from episode 3 on; owner decision 2026-10-19)
Every episode belongs to one of four named series, defined in `topics.json` → `series`:
- `numbers` **ORTHO NUMBERS**: one number every orthopod must know (thresholds, cut-offs, time limits)
- `classified` **CLASSIFIED**: a classification drawn type by type, with what changes management
- `rules` **CLINICAL RULES**: a validated clinical decision rule or criteria set
- `dontmiss` **DON'T MISS**: a can't-miss diagnosis: the signs, the test, the pitfall
Each topic in `topics.json` has a `series` key. **Part number** = (number of already-published episodes in that series) + 1,
computed at build time. Series let viewers binge and follow "for the next part".
- On screen: the hook pill reads `<SERIES TAG>  #<part>` (e.g. `CLINICAL RULES  #1`) instead of `ORTHO TIP #NN`.
- Caption: right after the hook line add `🧩 <Series name> · Part <part> — follow for the next one`.
- TikTok title: `<Series name> #<part> · <short title>`. YouTube title: `<short title> | <Series name> #<part> #Shorts`.
- New topics you add must get a `series`. Keep the queue rotating between series (never 3 of the same series in a row).

## Visual themes (owner decision 2026-10-23)
- Brand constants never change: navy base, logo, Montserrat, the logo sting and outro.
- `kit/ep_common.py` sets the accent colour (the `SKY` name every script uses) from the episode's series automatically:
  numbers = sky, rules = gold, classified = violet, dontmiss = rose. Always use `SKY` (never hard-code a colour) for accents.
- The background style + camera move change every 5 episodes (~weekly): stars, grid, waves, aurora (`THEME_BG`).
- Green GOOD / coral WARN keep their right-vs-wrong meaning in every theme.
- Override for tests: ORTHO_SERIES=..., ORTHO_BG=0..3 before `script.py still ...`.

## Quality bar (from episode 3 on — the owner wants best quality)
- **Richer visuals are mandatory.** Every episode must contain at least one real, animated, anatomically accurate diagram as its centrepiece, not mostly text cards. Examples:
  - a bone or joint drawn as an X-ray, like `kit/femur.py`
  - a cross-section, like the leg compartments in `episodes/ep02_acs`
  - a classification drawn type by type (Salter-Harris I–V, Gartland, Schatzker, Garden)
  - a clinical test shown step by step with a simple limb outline
  - a measurement drawn on the image, like TAD
- **Animate the diagram so it explains something:**
  - structures build in
  - labels point to them
  - the abnormal part is highlighted
  - the measurement is drawn
  - good vs bad is compared side by side
- Text cards support the diagram; they don't replace it.
- Draw the anatomy from known proportions and landmarks, check it on stills, and fix anything wrong before rendering. Simplified schematics are fine if they are labelled and correct.
- **Design review gate (mandatory, before the full render).**
  - Render one still per scene, plus the cover frame. Give them to a separate reviewer agent that has not seen your code.
  - It must PASS each still on all of these:
    - no overlapping or cut-off text
    - all text at least 27 px and readable on a phone
    - no tofu or missing-glyph boxes
    - content inside x 60–960 and y 200–1700
    - anatomy correct and labels pointing to the right structures
    - a centrepiece diagram is present
    - brand colours used
    - the cover frame works as a thumbnail
  - Fix every FAIL and re-check until all pass.
- **Sharpest file.** `tools/build_episode.py` encodes the upload copy with x264 `-preset slow -tune animation` at about **10 Mbps**: roughly 2x the old quality, at 1080x1920, 30 fps, AAC 192k.
  - The bitrate is capped automatically so the file stays under ~90 MB, because GitHub's per-file limit is 100 MB.
  - Do not lower these settings.
  - The encode takes about 5–6 min on top of the render.

## Run steps
1. **Pick the topic.**
   - Read `insights.md` first, if it exists. It is the weekly performance report: what topics, hooks and formats did best.
   - When the queue offers a choice, or when you add new topics, favour what performs best. Keep the medical value high and keep rotating areas.
   - Open `topics.json`. If an item has status `ready`, publish that one: skip to step 6 using its existing render, or rebuild it if `out/` is empty.
   - Otherwise take the first item with status `todo`.
   - **Keep the queue topped up.** If fewer than 5 items have status `todo`, first append 10 new topics to `topics.json` (status `todo`, the next episode numbers and slugs, and a `source_hint` naming the primary paper(s)).
     - Pick high-yield, practical orthopedic topics that have not been covered yet.
     - Rotate areas: trauma, paediatrics, spine, sports, hand, arthroplasty, oncology, foot & ankle, emergency orthopedics.
     - Each topic needs a clear rule, number, classification or test, backed by a findable primary source.
     - Never repeat a slug or title already in the file.
2. **Research.**
   - Use the PubMed connector to find the primary source(s) named in `source_hint`. Read the abstract and metadata.
   - Collect only figures that appear in the source. Record PMID and DOI.
   - If PubMed is unavailable or the key claim can't be confirmed, set status `blocked` with a reason, push the change, notify the owner, and stop. Never publish unverified medical content.
3. **Write the episode.**
   - Copy `episodes/ep02_acs/acs.py` as the template to `episodes/epNN_slug/script.py`.
   - It must end with `main_cli(TL, SCENES, hook_end=<script s>, resume=<script s>)` so `script.py dur`, `seg` and `hookend` work.
   - The hook scene doubles as the TikTok **cover**. Design it like a thumbnail:
     - The series tag pill, e.g. `ORTHO NUMBERS  #3` (see Series).
     - One big, short headline: a number or 2–4 words.
     - A one-line subtitle saying why it matters.
     - Everything centred between y 400 and 1300, fully on screen by about 1.5 s.
   - Put the sources and verified figures in the module docstring.
4. **Check stills before the full render** (then run the design review gate described under Quality bar).
   - Run `python3 script.py still <t> ...` at one real time per scene, using `TL.real(script_t)`.
   - Look at every still. Fix any overlapping or cut-off text, unreadable small text, wrong anatomy, or missing glyph boxes.
5. **Render.**
   - Run `python3 tools/build_episode.py episodes/epNN_slug/script.py epNN_slug` in the background with a long timeout. It takes about 15–20 min.
   - Poll for `out/epNN_slug_1080p.mp4`. Do not use `pgrep -f` with a pattern that matches your own wait loop.
   - Verify the output:
     - Duration is about 60–70 s.
     - There is an audio stream; max volume is −91 dB across the body and present in the intro and outro.
     - Frames at 2 s, mid-body and the outro look right.
6. **Independent fact check (mandatory).**
   - Spawn a separate agent that has not seen your work.
   - Give it the episode's list of on-screen claims and numbers, plus the PMIDs.
   - It must confirm each claim against PubMed and return PASS or FAIL per claim.
   - On any FAIL, fix the claim and re-render, or set the topic to `blocked`. Never publish a FAIL.
7. **Host the video.** Run `tools/publish_github.sh epNN_slug out/epNN_slug_1080p.mp4`.
   - It force-pushes the MP4 to the orphan branch `media` and prints the public raw URL.
   - GitHub Releases are not permitted for Claude sessions, so a branch is used instead.
   - The `media` branch only ever holds the current video; Metricool copies the file to its own storage when the post is created.
8. **Schedule on TikTok, and cross-post to Instagram Reels and YouTube Shorts when they are connected,** with the Metricool connector `createScheduledPost`.
   - **YouTube gets its own post with the music-free file** (owner decision 2026-10-20): YouTube Content ID claimed the Pixabay
     tracks (track01, 02, 03) and blocked EP01 and EP02 worldwide. So:
     a) host `out/<slug>_1080p.mp4` (with music) with publish_github.sh and create ONE post for **tiktok + instagram**;
     b) then host `out/<slug>_youtube_1080p.mp4` (original Orthojets theme from the music repo's `youtube_safe/` instead of Pixabay music,
        logo chime + outro sound; NO motion sound effects — owner decision 2026-10-20) with
        `tools/publish_github.sh <slug>_yt out/<slug>_youtube_1080p.mp4` and create a SECOND post for **youtube only**, same time,
        same caption, same youtubeData. (Do (a) fully before (b): the media branch keeps only the latest file, and Metricool copies the
        file when the post is created.)
     Never put a Pixabay track on YouTube. Only `youtube_safe/` tracks (YouTube Audio Library downloads, `yal_*`) may be used there — and NEVER use the
     `youtube_safe/` ensemble themes (yt1..yt4) on TikTok or Instagram (owner rule 2026-10-20). The original Steinway piano themes
     (`piano_*.mp3`, tools/piano_theme.py) are approved for every platform (owner 2026-10-21) and sit in both rotations.
   - Call `getBrandSettings` first. `networksData` lists the connected networks: TikTok always; Instagram and YouTube once the owner connects them in Metricool.
   - TikTok + Instagram share one post: `providers: [{"network":"tiktok"}, {"network":"instagram"}]`; YouTube is the separate post above.
     - Instagram: `instagramData: {"type":"REEL","showReelOnFeed":true}`
     - YouTube: `youtubeData: {"title":"<≤100 chars, ends with #Shorts>","type":"short","privacy":"public","madeForKids":false,"category":"EDUCATION","tags":[...]}`
     - If a network rejects the post, schedule TikTok alone and mention the error in the report.
   - First call `getScheduledPosts` for the target day. If a TikTok post is already scheduled that day, do not add a second one: move this episode to the next free Sun–Thu 10:00 slot (never Friday or Saturday).
   - Then create the post:
   - blogId `7507086`, timezone `Africa/Tripoli`
   - date: next day 10:00 (`YYYY-MM-DDT10:00:00+02:00`)
   - info:
     - `providers: [{"network":"tiktok"}]`, `autoPublish: true`
     - `media: [<raw URL from step 7>]`
     - `videoCoverMilliseconds`: the frame where the hook is fully on screen. With the hook-first structure that is `(hook_end − 0.4) × 1000` ms, about 2200–2600; `build_episode.py` prints it as "cover frame".
       - Never use a frame inside the logo flash: it shows the same logo for every episode.
       - Extract that frame with ffmpeg and look at it before scheduling. The tag, big number/title and subtitle must all be visible and sharp.
     - `text`: the English caption, in this order:
       - hook line
       - the rule
       - 3 ✅ tips
       - 📚 reference
       - "Save it & share it with a colleague 👨‍⚕️"
       - a blank line, then **one engagement question**: "💬 <a specific, easy-to-answer question that invites experience or opinion from doctors and students> 👇"
       - a blank line, then EXACTLY 5 hashtags (Instagram rejects/limits posts with more than 5 since 2026; the same caption goes to every network):
         #orthopedics #orthopedicsurgery <1 topic tag, the most-searched term, e.g. #hipfracture> #medtok #orthojets
         Never more than 5. Put extra topic keywords into the YouTube `tags` list instead (those are hidden and unlimited).
         Never use generic tags (#fyp #foryou #viral #trending #explore #creatorsearchinsights): no proven reach effect, they dilute the
         niche signal and waste a slot. Put the main search keywords in the caption's first line instead (TikTok search reads caption text).
         Owner decision 2026-10-19: the owner delegated this choice; keep this rule.
     - **No `<` or `>` anywhere in the caption, tiktok title or YouTube title/tags** (YouTube rejects the upload with
       "invalid or empty video title" — happened on EP01 2026-10-19). Write them as words: "under/below 25 mm", "over/above 38.5 °C",
       "at least" for ≥. (On-screen video text may still use < and >.) Also avoid `→` in captions; use a comma or "so".
       After scheduling, if any provider later shows status ERROR, fix and reschedule that network alone as a separate post.
     - `firstCommentText`: the same engagement question. Metricool posts it as the first comment on Instagram and YouTube. TikTok doesn't allow third-party comments, which is why the question is also in the caption.
     - `tiktokData: {"title":"Ortho Tip #NN · <short title>","privacyOption":"PUBLIC_TO_EVERYONE","disableComment":false,"disableDuet":false,"disableStitch":false,"autoAddMusic":false,"isAigc":false}`
     - `title` is required by Metricool.
     - `isAigc` is false because the videos are motion graphics rendered with code. Set it to true if TikTok's policy requires it for this content.
9. **Record.**
   - In `topics.json`, set the item to `published` with the date, release tag, Metricool post id/plannerUrl, PMIDs and `part`.
   - Also add a `carousel` block for the weekly Instagram carousel (only verified on-screen claims, no new facts):
     `{"headline": "<big number/title, ≤ 14 chars>", "sub": "<one line, ≤ 40 chars>", "points": ["<3 short claims, ≤ 80 chars each>"], "ref": "<Author et al., Journal Year>"}`
   - Commit `episodes/`, `topics.json` and any kit fixes. Never commit `out/`.
   - Push.
10. **Report.** Send the owner one short message, in Arabic: the episode title, when it goes live, the Metricool planner link, and the sources (PubMed attribution plus DOI links).

## Weekly carousel (separate scheduled task, Saturday evening; owner decision 2026-10-19)
1. Pick every episode whose video goes live in the 7 days ending the coming Sunday (from `topics.json`), max 5.
   If an episode lacks a `carousel` block, write it from its on-screen claims.
2. `python3 tools/build_carousel.py out/carousel_<YYYYMMDD> <slug> ... --week "<Mon d - Mon d>"` → cover + one slide per
   episode + follow slide (1080x1350 JPG). Look at every slide: no overlap, nothing cut off, glyphs fine.
3. `tools/publish_dir.sh carousel_<YYYYMMDD> out/carousel_<YYYYMMDD>` → one raw URL per slide (in order).
4. Metricool `createScheduledPost` for **Sunday 18:00** Africa/Tripoli: providers Instagram (`instagramData: {"type":"POST"}`)
   plus TikTok as a photo post when connected (`tiktokData` with `title` "This week's ortho pearls", `photoCoverIndex: 0`,
   `autoAddMusic: true`); `media` = the slide URLs in order. If TikTok rejects photo posts, schedule Instagram alone.
   Caption: `This week's ortho pearls 🦴 Save it for your next shift 🔖` + one line per episode (`✅ <headline> — <sub>`) +
   `📚 References on each slide` + `💬 Which one did you not know? 👇` + the 5 hashtags (#orthopedics #orthopedicsurgery
   #orthopaedics #medtok #orthojets). Skip the week if there are no episodes.
5. Record it under `carousels` in `topics.json`, push, and send the owner a short Arabic note.

## Failure handling
- Render or upload fails: retry once. If it fails again, leave the topic as `todo`, notify the owner, and stop.
- Metricool rejects the post: notify the owner with the exact error. Do not alter the medical text to work around it.
