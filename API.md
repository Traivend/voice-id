# Voice ID API Documentation

Base URL: `https://your-domain.com` oder `http://localhost:8088`

## Authentifizierung

Alle Endpoints erfordern den Header:
```
x-api-key: <API_KEY>
```

---

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### Sprecher registrieren (Enroll)

```
POST /enroll
```

Registriert einen neuen Sprecher oder aktualisiert einen bestehenden.

**Query Parameter:**
| Parameter | Typ | Required | Beschreibung |
|-----------|-----|----------|--------------|
| speaker_key | string | ✓ | Eindeutiger Identifier für den Sprecher |
| display_name | string | | Anzeigename (optional) |

**Body:** `multipart/form-data`
| Field | Typ | Beschreibung |
|-------|-----|--------------|
| file | file | Audio/Video Datei (WAV, MP3, MP4, etc.) |

**Empfehlung:** 60-120 Sekunden sauberes Audio für beste Ergebnisse.

**Request Beispiel:**
```bash
curl -X POST "https://api.example.com/enroll?speaker_key=markus&display_name=Markus%20M" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@recording.mp4"
```

**Response (neu):**
```json
{
  "message": "Speaker enrolled",
  "speaker_key": "markus"
}
```

**Response (aktualisiert):**
```json
{
  "message": "Speaker updated",
  "speaker_key": "markus"
}
```

---

### Sprecher identifizieren (Identify)

```
POST /identify
```

Identifiziert den Sprecher in einer Audio/Video Datei.

**Query Parameter:**
| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| threshold | float | 0.80 | Similarity-Schwellwert (0.0 - 1.0) |

**Threshold Tuning:**
- `0.76 - 0.78`: Wenn zu viele "unknown" Ergebnisse
- `0.80`: Standard
- `0.85+`: Wenn falsche Matches auftreten

**Body:** `multipart/form-data`
| Field | Typ | Beschreibung |
|-------|-----|--------------|
| file | file | Audio/Video Datei (WAV, MP3, MP4, etc.) |

**Request Beispiel:**
```bash
curl -X POST "https://api.example.com/identify?threshold=0.80" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@unknown_speaker.wav"
```

**Response (Match gefunden):**
```json
{
  "identified": true,
  "speaker_key": "markus",
  "display_name": "Markus M",
  "similarity": 0.9234
}
```

**Response (kein Match):**
```json
{
  "identified": false,
  "message": "No match above threshold",
  "best_match": {
    "speaker_key": "markus",
    "similarity": 0.7123
  }
}
```

**Response (keine Sprecher registriert):**
```json
{
  "identified": false,
  "message": "No speakers enrolled"
}
```

---

### Alle Sprecher auflisten

```
GET /speakers
```

**Request Beispiel:**
```bash
curl "https://api.example.com/speakers" \
  -H "x-api-key: YOUR_API_KEY"
```

**Response:**
```json
{
  "speakers": [
    {
      "speaker_key": "markus",
      "display_name": "Markus M",
      "created_at": "2024-01-15T10:30:00+00:00"
    },
    {
      "speaker_key": "anna",
      "display_name": null,
      "created_at": "2024-01-16T14:20:00+00:00"
    }
  ]
}
```

---

### Sprecher löschen

```
DELETE /speakers/{speaker_key}
```

**Path Parameter:**
| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| speaker_key | string | Identifier des Sprechers |

**Request Beispiel:**
```bash
curl -X DELETE "https://api.example.com/speakers/markus" \
  -H "x-api-key: YOUR_API_KEY"
```

**Response:**
```json
{
  "message": "Speaker deleted",
  "speaker_key": "markus"
}
```

**Error (nicht gefunden):**
```json
{
  "detail": "Speaker not found"
}
```

---

## Unterstützte Dateiformate

| Format | Typ |
|--------|-----|
| WAV, FLAC, OGG, AIFF | Audio (direkt) |
| MP3, M4A, AAC | Audio (via ffmpeg) |
| MP4, MKV, MOV, WebM | Video (via ffmpeg) |

---

## Error Responses

**401 Unauthorized:**
```json
{
  "detail": "Invalid API key"
}
```

**400 Bad Request:**
```json
{
  "detail": "Failed to process audio: <error message>"
}
```

**404 Not Found:**
```json
{
  "detail": "Speaker not found"
}
```
