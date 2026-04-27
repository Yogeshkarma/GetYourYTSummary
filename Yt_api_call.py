from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

class Youtube_Fetcher:
    def __init__(self,url_id):
        self.url_id=url_id
        self.transcript=None

    def url_validator(self,url_id):
        if "youtube.com" in self.url_id or "youtu.be" in self.url_id:
            if "watch?v=" in self.url_id:
                return self.url_id.split("watch?v=")[-1].split("&")[0]
            elif "youtu.be/" in self.url_id:
                return self.url_id.split("youtu.be/")[-1].split("?")[0]
        return self.url_id

    def fetch_transcript(self,url_id):
        self.transcript=YouTubeTranscriptApi().fetch(self.url_id,languages=[en,hi])
        self.transcript = " ".join([chunk.get("text", "") for chunk in transcript_list])
        return self.transcript



