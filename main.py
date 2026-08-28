from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from difflib import SequenceMatcher
from typing import Any, Literal

import httpx
import yt_dlp
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from yt_dlp.utils import DownloadError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("searchtracks")

# ---------------------------------------------------------------------------
# Tidal credentials + search
# ---------------------------------------------------------------------------

TIDAL_CLIENT_ID = "drXbs9hIKPhodGlt"
TIDAL_CLIENT_SECRET = "NUB8jAxu4szFRNpKe4o4LgLVuVweNBUK6ziYz5CQZss="
TIDAL_AUTH_URL = "https://auth.tidal.com/v1/oauth2/token"
TIDAL_SEARCH_URL = "https://openapi.tidal.com/v2/searchResults"
TIDAL_TRACKS_URL = "https://openapi.tidal.com/v2/tracks"


async def get_tidal_access_token() -> str:
    """Requests a brand new access token from Tidal on every call, no caching."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            TIDAL_AUTH_URL,
            data={"grant_type": "client_credentials"},
            auth=(TIDAL_CLIENT_ID, TIDAL_CLIENT_SECRET),
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to obtain a Tidal access token: {response.text}",
        )

    token = response.json().get("access_token")
    if not token:
        raise HTTPException(status_code=502, detail="Tidal's token response had no access_token")
    return str(token)


async def tidal_search(query: str, country_code: str, include: str) -> dict[str, Any]:
    token = await get_tidal_access_token()

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            TIDAL_SEARCH_URL,
            params={
                "countryCode": country_code,
                "filter[query]": query,
                "include": include,
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.api+json",
            },
        )

    if response.status_code >= 400:
        status = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status, detail=response.text)

    return response.json()


async def get_tidal_track(track_id: str, country_code: str) -> dict[str, Any]:
    """Fetches a single track by id, with its artists and albums included, so the id from a
    search result's tracks/topHits relationship can be turned into a title + artist name."""
    token = await get_tidal_access_token()

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{TIDAL_TRACKS_URL}/{track_id}",
            params={"countryCode": country_code, "include": "artists,albums"},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.api+json",
            },
        )

    if response.status_code >= 400:
        status = response.status_code if response.status_code < 500 else 502
        raise HTTPException(status_code=status, detail=response.text)

    return response.json()


def parse_tidal_track(payload: dict[str, Any]) -> dict[str, Any]:
    """Pulls title, artist name(s) and album title out of a /tracks/{id} response."""
    data = payload.get("data") or {}
    attributes = data.get("attributes") or {}
    relationships = data.get("relationships") or {}
    included = payload.get("included") or []

    title = str(attributes.get("title") or "").strip()
    version = attributes.get("version")
    if version:
        title = f"{title} ({version})".strip()

    artist_ids = {item.get("id") for item in (relationships.get("artists", {}).get("data") or [])}
    album_ids = {item.get("id") for item in (relationships.get("albums", {}).get("data") or [])}

    artist_names: list[str] = []
    album_title: str | None = None
    for item in included:
        item_type = item.get("type")
        item_id = item.get("id")
        item_attributes = item.get("attributes") or {}
        if item_type == "artists" and item_id in artist_ids:
            name = item_attributes.get("name")
            if name:
                artist_names.append(str(name))
        elif item_type == "albums" and item_id in album_ids and album_title is None:
            album_title = item_attributes.get("title")

    return {
        "title": title,
        "artist": ", ".join(artist_names),
        "album": album_title,
    }


# ---------------------------------------------------------------------------
# Local searchtracks.json catalog (title/artist -> cdn_link)
# ---------------------------------------------------------------------------

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "searchtracks.json")
LOCAL_MATCH_MIN_SCORE = 0.72

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(value: str | None) -> str:
    return _NORMALIZE_RE.sub(" ", (value or "").lower()).strip()


def load_catalog(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        logger.warning("Catalog file not found at %s, local CDN lookups will always miss.", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tracks: list[dict[str, Any]] = []
    for entry in raw_data if isinstance(raw_data, list) else []:
        if not isinstance(entry, dict):
            continue
        cdn_link = entry.get("cdn_link")
        title = entry.get("Title") or entry.get("title")
        artist = entry.get("Artist") or entry.get("artist")
        if not cdn_link or not title or not artist:
            continue
        tracks.append(
            {
                "title": str(title),
                "artist": str(artist),
                "album": entry.get("Album") or entry.get("album"),
                "duration": entry.get("Duration") or entry.get("duration"),
                "thumbnail": entry.get("Thumbnail") or entry.get("thumbnail"),
                "cdn_link": str(cdn_link),
            }
        )

    logger.info("Loaded %d playable tracks from %s", len(tracks), path)
    return tracks


CATALOG = load_catalog(CATALOG_PATH)


def find_local_track(title: str, artist: str) -> dict[str, Any] | None:
    target_title = normalize(title)
    target_artist = normalize(artist)
    if not target_title:
        return None

    best_track: dict[str, Any] | None = None
    best_score = 0.0

    for track in CATALOG:
        title_score = SequenceMatcher(None, target_title, normalize(track["title"])).ratio()
        if title_score < 0.5:
            continue
        artist_score = SequenceMatcher(None, target_artist, normalize(track["artist"])).ratio()
        score = (title_score * 0.7) + (artist_score * 0.3)
        if score > best_score:
            best_score = score
            best_track = track

    if best_track and best_score >= LOCAL_MATCH_MIN_SCORE:
        return best_track
    return None


async def find_tidal_id_and_album(title: str, artist: str, country_code: str = "US") -> tuple[str | None, str | None]:
    """Used only to backfill ID/Album before logging to top_tracks.json when a plain
    title/artist /api/v1/play call didn't already have them. Searches Tidal for the closest
    matching track, then fetches that track's full detail to read off its id and album."""
    try:
        payload = await tidal_search(query=f"{title} {artist}", country_code=country_code, include="tracks")
    except HTTPException:
        return None, None

    result_data = payload.get("data") or []
    if not result_data:
        return None, None

    candidate_ids = [
        ref.get("id")
        for ref in ((result_data[0].get("relationships") or {}).get("tracks", {}).get("data") or [])
        if ref.get("id")
    ]
    if not candidate_ids:
        return None, None

    track_by_id = {
        item["id"]: item for item in (payload.get("included") or []) if item.get("type") == "tracks"
    }

    target_title = normalize(title)
    best_id = candidate_ids[0]
    best_score = 0.0
    for candidate_id in candidate_ids:
        candidate = track_by_id.get(candidate_id)
        if not candidate:
            continue
        candidate_title = (candidate.get("attributes") or {}).get("title") or ""
        score = SequenceMatcher(None, target_title, normalize(candidate_title)).ratio()
        if score > best_score:
            best_score = score
            best_id = candidate_id

    try:
        track_payload = await get_tidal_track(best_id, country_code)
    except HTTPException:
        return best_id, None

    details = parse_tidal_track(track_payload)
    return best_id, details.get("album")


# ---------------------------------------------------------------------------
# top_tracks.json: tracks that had no cdn_link and got resolved via YouTube.
# Logged with cdn_link left empty, so a separate upload job can fill it in later.
# ---------------------------------------------------------------------------

TOP_TRACKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "top_tracks.json")
_top_tracks_lock = threading.Lock()


def _load_top_tracks(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not parse %s as JSON, starting fresh.", path)
            return []
    return data if isinstance(data, list) else []


TOP_TRACKS: list[dict[str, Any]] = _load_top_tracks(TOP_TRACKS_PATH)
TOP_TRACKS_SEEN: set[str] = set()
for _entry in TOP_TRACKS:
    if isinstance(_entry, dict):
        _title = _entry.get("Title") or _entry.get("title")
        _artist = _entry.get("Artist") or _entry.get("artist")
        if _title and _artist:
            TOP_TRACKS_SEEN.add(f"{normalize(_title)}|{normalize(_artist)}")


def format_duration(seconds: float | None) -> str | None:
    """Matches searchtracks.json's Duration format, e.g. 03:42 (or 1:03:42 past an hour)."""
    if seconds is None:
        return None
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def record_top_track(
    track_id: str | None,
    title: str,
    artist: str,
    album: str | None,
    duration: float | None,
    thumbnail: str | None,
    youtube_url: str,
) -> None:
    """Blocking: run in a thread. Appends a title/artist not found locally to top_tracks.json,
    in the same field shape as searchtracks.json, with cdn_link left empty until a separate
    job downloads and uploads it to the CDN. Skips it if that title/artist is already logged."""
    key = f"{normalize(title)}|{normalize(artist)}"
    with _top_tracks_lock:
        if key in TOP_TRACKS_SEEN:
            return
        TOP_TRACKS_SEEN.add(key)
        TOP_TRACKS.append(
            {
                "ID": track_id,
                "Title": title,
                "Artist": artist,
                "Album": album,
                "Duration": format_duration(duration),
                "Thumbnail": thumbnail,
                "youtube_url": youtube_url,
                "cdn_link": "",
            }
        )
        os.makedirs(os.path.dirname(TOP_TRACKS_PATH), exist_ok=True)
        with open(TOP_TRACKS_PATH, "w", encoding="utf-8") as f:
            json.dump(TOP_TRACKS, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# YouTube fallback: official Data API for search, yt-dlp only to pull the stream
# ---------------------------------------------------------------------------

YOUTUBE_API_KEY = "AIzaSyD68Yxqvc-PNNaCYTjdbekzdFG9QOoYR40"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_SEARCH_COUNT = 5
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _match_score(query_title: str, query_artist: str, candidate_title: str, candidate_channel: str) -> float:
    title_score = SequenceMatcher(None, normalize(query_title), normalize(candidate_title)).ratio()
    artist_in_title = normalize(query_artist) in normalize(candidate_title)
    artist_score = SequenceMatcher(None, normalize(query_artist), normalize(candidate_channel)).ratio()
    return title_score * 0.6 + (1.0 if artist_in_title else artist_score) * 0.4


async def youtube_search(query: str, max_results: int = YOUTUBE_SEARCH_COUNT) -> list[dict[str, Any]]:
    """Uses the official YouTube Data API to find candidate videos for a title/artist query."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            YOUTUBE_SEARCH_URL,
            params={
                "key": YOUTUBE_API_KEY,
                "q": query,
                "part": "snippet",
                "type": "video",
                "maxResults": max_results,
            },
        )

    if response.status_code >= 400:
        raise RuntimeError(f"YouTube search failed: {response.text}")

    items = response.json().get("items", [])
    results: list[dict[str, Any]] = []
    for item in items:
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet") or {}
        if not video_id:
            continue
        thumbnails = snippet.get("thumbnails") or {}
        thumbnail = (thumbnails.get("high") or thumbnails.get("default") or {}).get("url")
        results.append(
            {
                "id": video_id,
                "title": snippet.get("title") or "",
                "channel": snippet.get("channelTitle") or "",
                "thumbnail": thumbnail,
            }
        )
    return results


def _ydl_options() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "geo_bypass": True,
    }


def _pick_audio_format(info: dict[str, Any]) -> dict[str, Any] | None:
    formats = info.get("formats")
    if isinstance(formats, list) and formats:
        playable = [f for f in formats if f.get("url")]
        audio_only = [f for f in playable if f.get("acodec") not in (None, "none") and f.get("vcodec") in (None, "none")]
        pool = audio_only or playable
        if pool:
            return max(pool, key=lambda f: f.get("abr") or 0)
    if info.get("url"):
        return info
    return None


def extract_stream(video_id: str, fallback_title: str, fallback_thumbnail: str | None) -> dict[str, Any]:
    """Blocking: run in a thread. Pulls a playable audio stream URL for an already-chosen video."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    with yt_dlp.YoutubeDL({**_ydl_options(), "format": "bestaudio/best"}) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
        except DownloadError as error:
            raise RuntimeError(_ANSI_ESCAPE.sub("", str(error)).strip()) from error

    if not isinstance(info, dict):
        raise RuntimeError(f"yt-dlp returned no video information for '{video_url}'")

    chosen = _pick_audio_format(info)
    if not chosen:
        raise RuntimeError("yt-dlp did not return a playable audio stream for this video")

    protocol = str(chosen.get("protocol") or "").lower()
    stream_url = str(chosen["url"])
    is_hls = "m3u8" in protocol or stream_url.split("?", 1)[0].endswith(".m3u8")

    return {
        "stream_type": "hls" if is_hls else "progressive",
        "url": stream_url,
        "youtube_url": str(info.get("webpage_url") or video_url),
        "youtube_title": str(info.get("title") or fallback_title or ""),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail") or fallback_thumbnail,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class PlayableTrack(BaseModel):
    source: Literal["cdn", "youtube"] = Field(
        ..., description="cdn = matched in the local searchtracks catalog, youtube = resolved live via yt-dlp."
    )
    stream_type: Literal["progressive", "hls"] = Field(
        ..., description="progressive = direct file/stream URL, hls = an .m3u8 manifest URL."
    )
    url: str = Field(..., description="The playable audio URL to hand to a player.")
    title: str
    artist: str
    album: str | None = None
    duration: float | str | None = None
    thumbnail: str | None = None
    youtube_url: str | None = Field(None, description="Set only when source is youtube.")
    matched_youtube_title: str | None = None
    warning: str | None = None


app = FastAPI(
    title="Tidal Search and Play API",
    description=(
        "Wraps Tidal's search endpoint and resolves a chosen track to a playable audio URL: "
        "first from a local CDN catalog (data/searchtracks.json), and if it is not there, by "
        "searching YouTube for the closest match and extracting a stream URL with yt-dlp."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["system"], summary="Health check")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "catalog_tracks_loaded": len(CATALOG),
        "top_tracks_logged": len(TOP_TRACKS),
    }


@app.get(
    "/api/v1/search",
    tags=["search"],
    summary="Search the Tidal catalog",
    description="Proxies Tidal's /v2/searchResults endpoint, fetching a fresh access token on every call.",
)
async def search(
    query: str = Query(..., min_length=1, description="Free text search, for example an artist or track name."),
    country_code: str = Query("US", alias="countryCode", description="ISO country code Tidal should search in."),
    include: str = Query(
        "albums,artists,tracks,playlists,topHits,videos",
        description="Comma separated resource types to include, matching Tidal's own include parameter.",
    ),
) -> dict[str, Any]:
    return await tidal_search(query=query, country_code=country_code, include=include)


async def resolve_playable_track(
    title: str, artist: str, album: str | None, track_id: str | None = None
) -> PlayableTrack:
    """Shared by /api/v1/play and /api/v1/play-by-id: local catalog first, YouTube fallback second."""
    local_match = find_local_track(title, artist)
    if local_match:
        return PlayableTrack(
            source="cdn",
            stream_type="progressive",
            url=local_match["cdn_link"],
            title=local_match["title"],
            artist=local_match["artist"],
            album=local_match["album"],
            duration=local_match["duration"],
            thumbnail=local_match["thumbnail"],
        )

    try:
        candidates = await youtube_search(f"{title} {artist}".strip())
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    if not candidates:
        raise HTTPException(status_code=502, detail=f"No YouTube results found for '{title} {artist}'")

    best = max(candidates, key=lambda c: _match_score(title, artist, c["title"], c["channel"]))

    try:
        resolved = await asyncio.to_thread(extract_stream, best["id"], best["title"], best["thumbnail"])
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    logged_id, logged_album = track_id, album
    if logged_id is None or logged_album is None:
        found_id, found_album = await find_tidal_id_and_album(title, artist)
        logged_id = logged_id or found_id
        logged_album = logged_album or found_album

    await asyncio.to_thread(
        record_top_track,
        logged_id,
        title,
        artist,
        logged_album,
        resolved["duration"],
        resolved["thumbnail"],
        resolved["youtube_url"],
    )

    return PlayableTrack(
        source="youtube",
        stream_type=resolved["stream_type"],
        url=resolved["url"],
        title=title,
        artist=artist,
        album=logged_album,
        duration=resolved["duration"],
        thumbnail=resolved["thumbnail"],
        youtube_url=resolved["youtube_url"],
        matched_youtube_title=resolved["youtube_title"],
        warning=(
            "This URL is served directly from YouTube's CDN and is temporary, it can expire "
            "or be IP bound. Call this endpoint again if playback fails."
        ),
    )


@app.get(
    "/api/v1/play",
    response_model=PlayableTrack,
    tags=["play"],
    summary="Resolve a clicked track (by title/artist) to a playable audio URL",
    description=(
        "Takes the title and artist of a track the user clicked in the Tidal search results. "
        "Checks the local catalog (data/searchtracks.json) first and returns its cdn_link on a "
        "match. Otherwise searches YouTube (via the YouTube Data API) for the same title and "
        "artist, picks the closest matching video, and pulls a playable audio stream URL out "
        "of it with yt-dlp."
    ),
)
async def play(
    title: str = Query(..., min_length=1, description="Track title from the Tidal search result the user clicked."),
    artist: str = Query(..., min_length=1, description="Artist name from the same Tidal search result."),
    album: str | None = Query(None, description="Optional album name, passed through on a YouTube fallback."),
) -> PlayableTrack:
    return await resolve_playable_track(title, artist, album)


@app.get(
    "/api/v1/play-by-id",
    response_model=PlayableTrack,
    tags=["play"],
    summary="Resolve a clicked track (by Tidal track id) to a playable audio URL",
    description=(
        "Takes a Tidal track id straight from a /api/v1/search response (its data.relationships."
        "tracks or data.relationships.topHits entries with type 'tracks'). Looks that track up "
        "on Tidal to get its title and artist, then runs the same resolution flow as /api/v1/"
        "play: local CDN catalog first, YouTube fallback second."
    ),
)
async def play_by_id(
    track_id: str = Query(..., min_length=1, description="A Tidal track id, e.g. '173132309'."),
    country_code: str = Query("US", alias="countryCode", description="ISO country code to look the track up in."),
) -> PlayableTrack:
    payload = await get_tidal_track(track_id, country_code)
    details = parse_tidal_track(payload)

    if not details["title"] or not details["artist"]:
        raise HTTPException(
            status_code=502,
            detail=f"Tidal track {track_id} did not include a resolvable title and artist",
        )

    return await resolve_playable_track(
        details["title"], details["artist"], details["album"], track_id=track_id
    )
