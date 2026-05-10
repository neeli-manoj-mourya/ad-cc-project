"""Quick sync diagnostics from metadata."""
import json

d = json.load(open("output/thanu_short_film_ad_metadata.json", "r", encoding="utf-8"))
print(f"Clips: {d['total_clips']}")
print(f"Overflows: {d['overflow_clips']}")
print(f"Max drift: {d['max_drift_ms']}ms")
print()
for c in d["clips"]:
    print(
        f"  #{c['gap_id']:02d}  "
        f"target={c['target_timestamp']:8.3f}s  "
        f"actual={c['actual_timestamp']:8.3f}s  "
        f"drift={c['drift_ms']:+.4f}ms  "
        f"dur={c['audio_duration']:7.3f}s  "
        f"margin={c['margin']:+7.3f}s  "
        f"{c['status']}"
    )
