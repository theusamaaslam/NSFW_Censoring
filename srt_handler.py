import os
import json
import pysrt
import re

def time_to_ms(srt_time):
    return (
        srt_time.hours * 3600000 +
        srt_time.minutes * 60000 +
        srt_time.seconds * 1000 +
        srt_time.milliseconds
    )

def load_report(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trimmed_segments = [
        (int(seg["start_time_ms"]), int(seg["end_time_ms"]))
        for seg in data.get("trimmed_segments", [])
    ]

    nsfw_words = set()
    for word_info in data.get("nsfw_words", []):
        nsfw_words.update(word_info.get("words", []))

    return trimmed_segments, list(nsfw_words)

def overlaps(start_ms, end_ms, segments):
    for seg_start, seg_end in segments:
        if start_ms < seg_end and end_ms > seg_start:
            return True
    return False

def censor_text(text, word):
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    return pattern.sub(lambda m: '*' * len(m.group()), text)

def censor_srt_file(input_srt_path, output_srt_path, censorship_json_path):
    if not os.path.exists(input_srt_path):
        raise FileNotFoundError(f"❌ Input SRT not found: {input_srt_path}")
    if not os.path.exists(censorship_json_path):
        raise FileNotFoundError(f"❌ Censorship JSON not found: {censorship_json_path}")

    trimmed_segments, nsfw_words = load_report(censorship_json_path)
    subtitles = pysrt.open(input_srt_path)
    cleaned_subs = pysrt.SubRipFile()

    for sub in subtitles:
        start_ms = time_to_ms(sub.start)
        end_ms = time_to_ms(sub.end)

        # Remove subtitle if overlapping with trimmed segment
        if overlaps(start_ms, end_ms, trimmed_segments):
            print(f"✂️ Removed subtitle [{sub.index}] (trim overlap)")
            continue

        original_text = sub.text
        for word in nsfw_words:
            sub.text = censor_text(sub.text, word)

        if original_text != sub.text:
            print(f"🔤 Censored subtitle [{sub.index}]")

        cleaned_subs.append(sub)

    cleaned_subs.save(output_srt_path, encoding="utf-8")
    print(f"✅ Cleaned SRT saved to: {output_srt_path}")
    return output_srt_path

# Optional CLI usage
if __name__ == "__main__":
    input_file = "The Wolf Of Wall Street 2013 BDRip 1080p DTS-HighCode_track3_eng.srt"
    output_file = "cleaned_output.srt"
    json_file = "nsfw_detection_report.json"
    censor_srt_file(input_file, output_file, json_file)
