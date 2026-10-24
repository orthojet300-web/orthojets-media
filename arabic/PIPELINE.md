# «عظامك ببساطة» — Arabic explainer series for the general public (Saturday 10:00)

Owner decision 2026-10-24: one Arabic animated explainer every **Saturday 10:00 Africa/Tripoli**, aimed at the general
public (patients, parents, students), on TikTok, Instagram and YouTube. Built by the scheduled task on **Friday evening**.
The weekday English episodes (Sun–Thu, PIPELINE.md) are unchanged.

## Format (owner-approved, from the Growth Plate episode — keep)
- Flat cartoon explainer, LIGHT palette (sky-white gradient), navy `#223F61` + sky `#78AFD8`, coral/green only for wrong/right.
  Medical labels in English on screen; narration in Arabic. Small `orthojets` watermark at the bottom.
- Voice: Piper `ar_JO-kareem-medium`, Fusha with FULL tashkeel, `--length-scale 0.75`. Pause form (وقف) is applied by
  `tools/waqf.py` to the TTS text only. Captions: one word at a time (Cairo 900, 104 px, white with navy stroke, y = 445).
- Music: `tools/music.py` (our own soft pad + bells, no percussion) at -30 LUFS under the voice at -16. Same file on all
  three platforms (no third-party music -> no Content ID).
- Structure: hook question (curiosity + stakes) -> 1 s logo sting -> explanation built step by step with ONE animated
  diagram that evolves (camera moves, zooms, highlights) -> what it means for you -> practical lesson -> CTA
  (like / follow / share) -> 3 s outro. Length 70–85 s.
- Plain language for non-doctors; every number/claim verified on PubMed (refText() on screen); no treatment advice beyond
  "see a doctor if…". Description always ends with: "محتوى توعوي، لا يغني عن استشارة الطبيب".

## Files
- `tools/`: gen_audio.py (Piper), assemble.py (voice.wav + timeline.json with hook_end), music.py, render.py (Playwright,
  1080x1920, 30 fps), stills.py, waqf.py, build_ar.py (all steps + kit/add_brand.py + 10 Mbps encode).
- `episodes/arNN_slug/`: anim.html (scenes as functions of time; copy ar01 and replace the scenes), script.json
  (`[scene_key, diacritized_text]`, clauses split on `،` and `:`), voice/timeline (generated), cover.jpg.
- `topics.json`: the queue.

## Run steps (Friday task)
1. Take the first `todo` in `arabic/topics.json` (add 6 new public-interest topics when fewer than 4 remain).
2. PubMed research: confirm every claim; note PMIDs/DOIs in `episodes/<slug>/SOURCES.md`.
3. Write `script.json` in simple Fusha WITH FULL tashkeel (hook first; 15–20 segments; last = CTA like ar01).
4. Build `anim.html` from `episodes/ar01_growth_plate/anim.html`: keep the helpers (pill, callout, refText, captions,
   background, watermark, CTA scene, warp()/OB/SC timing), replace the scene drawings for the new topic. Anatomy must be correct.
5. `python3 arabic/tools/gen_audio.py EP && python3 arabic/tools/assemble.py EP`, then `stills.py EP <one time per scene>`;
   a separate reviewer agent must PASS the stills (correct anatomy, readable Arabic, nothing cut off, brand look).
6. `python3 arabic/tools/build_ar.py EP SLUG --skip-tts` (~12 min). Check duration, audio (voice audible over music), frames.
7. Independent fact-check agent: PASS per claim against PubMed, or fix / block.
8. Cover (owner 2026-10-24): `python3 arabic/tools/make_cover.py EP/st/<hook still>.png EP/cover.jpg N "<line 1>" "<line 2>"` —
   series badge «عظامك ببساطة» + «كل يوم سبت» with the episode number, the English topic pill, the two-line Arabic question, hook art. Host video + cover with tools/publish_dir.sh.
9. Metricool (brand 7507086): ONE post for tiktok + instagram (REEL) + youtube (short), Saturday 10:00, same video.
   - Use `videoThumbnailUrl` = cover for instagram/youtube when accepted; otherwise fall back to `videoCoverMilliseconds`.
   - Caption: Arabic hook question, 3 short Arabic points, `📚` refs, "احفظه وشاركه مع من يهمه الأمر 💙",
     `💬` a simple question for the public, the disclaimer line, then exactly 5 hashtags:
     `#عظامك_ببساطة #صحة #طب #orthojets #<topic tag>`. No `<` or `>`.
   - tiktokData.title / youtubeData.title (≤100 chars, Arabic, ends with #Shorts on YouTube).
10. Mark the topic `published` with ids; commit and push; short Arabic report to the owner.
