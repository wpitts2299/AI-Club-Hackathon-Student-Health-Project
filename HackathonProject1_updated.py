import argparse
import bisect
import csv
import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Disable Hugging Face telemetry so test inputs never leave the server.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
import html
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from fastapi import Depends, FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
import torch.nn.functional as F
from transformers.utils import logging

try:
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover - optional dependency
    Fernet = None

app = FastAPI(
    title="Hackathon Student Support API",
    description="Analyze student stress and emotional text inputs via REST.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("HACKATHON_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
therapist_key_header = APIKeyHeader(name="X-Therapist-Key", auto_error=False)
therapist_basic = HTTPBasic(auto_error=False)
logger = logging.get_logger(__name__)
MIN_RESPONSE_WORDS = int(os.getenv("HACKATHON_MIN_RESPONSE_WORDS", "50"))
FIRST_RESPONDER_CONTACT = "(470) 578-6666"
CUSTOM_DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Hackathon Student Support API</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: "Segoe UI", Arial, sans-serif;
            background: linear-gradient(135deg, #e0e7ff 0%, #f8fafc 100%);
            margin: 0;
            padding: 48px 16px 80px;
            color: #0f172a;
        }
        header {
            max-width: 900px;
            margin: 0 auto 24px;
            text-align: center;
        }
        header h1 { margin-bottom: 8px; font-size: 2.2rem; }
        header p { margin: 0; color: #475569; font-size: 1.05rem; }
        .opblock-card {
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 18px;
            box-shadow: 0 25px 60px rgba(15, 23, 42, 0.15);
            border-left: 10px solid #22c55e;
            overflow: hidden;
        }
        .opblock-summary {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 28px;
        }
        .endpoint-info {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 16px;
        }
        .method-pill {
            background: #22c55e;
            color: white;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 999px;
            letter-spacing: 0.08em;
        }
        .path-label {
            font-size: 1.4rem;
            font-weight: 600;
            color: #111827;
        }
        .try-button {
            background: #145da0;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 22px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        .try-button:disabled {
            opacity: 0.65;
            cursor: default;
        }
        .try-button:hover:not(:disabled) {
            background: #0f4c81;
        }
        .form-panel {
            display: none;
            border-top: 1px solid #e2e8f0;
            padding: 28px;
            background: #f8fafc;
        }
        .form-panel.active { display: block; }
        label { font-weight: 600; color: #0f172a; display: block; margin-bottom: 8px; }
        textarea {
            width: 100%;
            min-height: 180px;
            border: 1px solid #cbd5f5;
            border-radius: 12px;
            padding: 12px;
            font-size: 1rem;
            resize: vertical;
            background: white;
            color: #0f172a;
        }
        textarea:disabled {
            background: #e2e8f0;
            cursor: not-allowed;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #cbd5f5;
            border-radius: 10px;
            font-size: 0.95rem;
            margin-top: 4px;
        }
        .id-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .id-row input {
            flex: 1 1 220px;
        }
        .validate-btn {
            background: #0ea5e9;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 16px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        .validate-btn:hover:not(:disabled) { background: #0284c7; }
        .validate-btn:disabled { opacity: 0.65; cursor: default; }
        .validation {
            margin: 8px 0 16px;
            color: #475569;
            font-size: 0.95rem;
        }
        .status-ok { color: #15803d; }
        .status-error { color: #b91c1c; }
        .status-muted { color: #475569; }
        .actions {
            margin-top: 18px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .actions button {
            border: none;
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 1rem;
            cursor: pointer;
        }
        .execute-btn { background: #10b981; color: white; }
        .cancel-btn { background: #e2e8f0; color: #0f172a; }
        pre {
            background: #0f172a;
            color: #e2e8f0;
            padding: 16px;
            margin-top: 20px;
            border-radius: 12px;
            overflow-x: auto;
            max-height: 360px;
            white-space: pre-wrap;
        }
        @media (max-width: 640px) {
            .opblock-summary { flex-direction: column; align-items: flex-start; gap: 16px; }
        }
    </style>
</head>
<body data-min-words="__MIN_WORDS__">
    <header>
        <h1>Hackathon Student Support API</h1>
        <p>Securely analyze a student's text with a single POST request.</p>
    </header>
    <section class="opblock-card">
        <div class="opblock-summary">
            <div class="endpoint-info">
                <span class="method-pill">POST</span>
                <span class="path-label">/analyze</span>
            </div>
            <button class="try-button" id="tryButton">Try it out</button>
        </div>
        <div class="form-panel" id="formPanel">
            <label for="studentIdInput">Student ID</label>
            <div class="id-row">
                <input id="studentIdInput" type="text" placeholder="Enter your student ID to unlock the form" />
                <button type="button" class="validate-btn" id="validateButton">Validate ID</button>
            </div>
            <p id="validationStatus" class="validation status-muted">Enter your student ID to continue.</p>
            <label for="studentText">Student text</label>
            <textarea id="studentText" placeholder="I failed three classes and feel hopeless. I need help." disabled></textarea>
            <div style="margin:8px 0;">
                <label style="display:flex; align-items:center; gap:8px;">
                    <input id="consentCheckbox" type="checkbox" />
                    <span>I consent to the HIPAA notice.</span>
                </label>
            </div>
            <p id="wordCounter" class="validation status-muted" style="margin-top:6px;">0/__MIN_WORDS__ words (min).</p>
            <div id="extraCreditSection" style="display:none; margin-top:12px; background:#eef2ff; border-radius:12px; padding:12px;">
                <label for="extraCreditSelect">Choose a class to receive +1 extra credit</label>
                <select id="extraCreditSelect" disabled style="padding:10px; border-radius:10px; border:1px solid #cbd5f5; width:100%; background:#ffffff;">
                    <option value="">Validate your student ID first.</option>
                </select>
                <p id="extraCreditStatus" class="validation status-muted" style="margin-top:6px;">Select a class after validation.</p>
                <div id="extraCreditSummary" style="margin-top:8px; color:#334155; font-size:0.95rem;"></div>
            </div>
            <label for="apiKeyInput">Optional API key (X-API-Key header)</label>
            <input id="apiKeyInput" type="text" placeholder="Leave blank unless the server requires a key." />
            <div class="actions">
                <button type="button" class="execute-btn" id="executeButton" disabled>Execute</button>
                <button type="button" class="cancel-btn" id="cancelButton">Cancel</button>
            </div>
            <div id="resultPanel" style="display:none;">
                <p style="margin-top:20px;font-weight:600;">Response</p>
                <pre id="resultBody">Awaiting request...</pre>
                <p id="resourceLink" style="display:none; margin-top:18px; color:#0f172a;">
                    Need to talk to someone? Kennesaw State Counseling and Psychological Services can help:
                    <a href="https://campus.kennesaw.edu/current-students/student-affairs/wellbeing/counseling/index.php"
                       target="_blank" rel="noopener"
                       style="color:#145da0; font-weight:600;">
                        campus.kennesaw.edu/wellbeing/counseling
                    </a>
                    <br />
                    Want community? Find a club of friends through Kennesaw State Registered Student Organizations:
                    <a href="https://owllife.kennesaw.edu/"
                       target="_blank" rel="noopener"
                       style="color:#145da0; font-weight:600;">
                        owllife.kennesaw.edu
                    </a>
                    <br />
                    Having financial struggles? Don't use your money on groceries - use KSU Campus Pantry:
                    <a href="https://campus.kennesaw.edu/current-students/student-affairs/wellbeing/care-services/community-resources/campus-pantry.php"
                       target="_blank" rel="noopener"
                       style="color:#145da0; font-weight:600;">
                        campus.kennesaw.edu/wellbeing/care-services/community-resources/campus-pantry
                    </a>
                </p>
            </div>
        </div>
    </section>
    <script>
    (function(){
        const tryButton = document.getElementById('tryButton');
        const formPanel = document.getElementById('formPanel');
        const cancelButton = document.getElementById('cancelButton');
        const executeButton = document.getElementById('executeButton');
        const validateButton = document.getElementById('validateButton');
        const studentText = document.getElementById('studentText');
        const studentIdInput = document.getElementById('studentIdInput');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const validationStatus = document.getElementById('validationStatus');
        const resultPanel = document.getElementById('resultPanel');
        const resultBody = document.getElementById('resultBody');
        const resourceLink = document.getElementById('resourceLink');
        const extraCreditSection = document.getElementById('extraCreditSection');
        const extraCreditSelect = document.getElementById('extraCreditSelect');
        const extraCreditStatus = document.getElementById('extraCreditStatus');
        const extraCreditSummary = document.getElementById('extraCreditSummary');
        const consentCheckbox = document.getElementById('consentCheckbox');
        const wordCounter = document.getElementById('wordCounter');
        const MIN_WORDS = parseInt(document.body.dataset.minWords || "50", 10) || 50;
        let validatedStudent = null;
        let availableClasses = [];
        let extraCreditClaimed = false;

        const statusClasses = {
            ok: 'validation status-ok',
            error: 'validation status-error',
            muted: 'validation status-muted'
        };

        function setStatus(message, variant) {
            validationStatus.textContent = message;
            validationStatus.className = statusClasses[variant] || statusClasses.muted;
        }

        function formatOptionLabel(entry) {
            const points = typeof entry.extra_credit_points === 'number' ? entry.extra_credit_points : 0;
            return `${entry.name} (extra credit: ${points})`;
        }

        function resetExtraCredit() {
            availableClasses = [];
            extraCreditClaimed = false;
            extraCreditSelect.innerHTML = '<option value=\"\">Validate your student ID first.</option>';
            extraCreditSelect.disabled = true;
            extraCreditSection.style.display = 'none';
            extraCreditStatus.textContent = 'Select a class after validation.';
            extraCreditStatus.className = statusClasses.muted;
            extraCreditSummary.textContent = '';
        }

        function renderExtraCreditSummary() {
            if (!availableClasses.length) {
                extraCreditSummary.textContent = '';
                return;
            }
            const lines = availableClasses.map((c) => `${c.name}: ${c.extra_credit_points || 0} point${(c.extra_credit_points || 0) === 1 ? '' : 's'}`);
            extraCreditSummary.textContent = `Current totals — ${lines.join(' • ')}`;
        }

        function countWords(text) {
            if (!text) return 0;
            return text.trim().split(/\\s+/).filter(Boolean).length;
        }

        function updateWordCounter() {
            const count = countWords(studentText.value);
            wordCounter.textContent = `${count}/${MIN_WORDS} words (min)`;
            wordCounter.className = count >= MIN_WORDS ? statusClasses.ok : statusClasses.error;
            return count;
        }

        function updateExecuteState() {
            const count = updateWordCounter();
            const hasText = studentText.value.trim().length > 0;
            const unlocked = !!validatedStudent && count >= MIN_WORDS && hasText && consentCheckbox.checked;
            executeButton.disabled = !unlocked;
        }

        function populateExtraCreditOptions(classes) {
            availableClasses = classes || [];
            extraCreditSection.style.display = 'block';
            if (!availableClasses.length) {
                extraCreditSelect.innerHTML = '<option value=\"\">No classes found on your roster.</option>';
                extraCreditSelect.disabled = true;
                extraCreditStatus.textContent = 'No classes available for extra credit.';
                extraCreditStatus.className = statusClasses.error;
                renderExtraCreditSummary();
                return;
            }

            extraCreditSelect.disabled = false;
            extraCreditSelect.innerHTML = '';
            availableClasses.forEach((course) => {
                const option = document.createElement('option');
                option.value = course.key;
                option.textContent = formatOptionLabel(course);
                extraCreditSelect.appendChild(option);
            });
            extraCreditStatus.textContent = 'Pick a class to receive +1 extra credit when you submit.';
            extraCreditStatus.className = statusClasses.muted;
            renderExtraCreditSummary();
        }

        function updateClassOptionLabel(key, newTotal) {
            const match = availableClasses.find((item) => item.key === key);
            if (match) {
                match.extra_credit_points = newTotal;
            }
            if (!availableClasses.length) {
                return;
            }
            // Refresh the visible labels without rebuilding the dropdown to avoid hiding it
            Array.from(extraCreditSelect.options).forEach((opt) => {
                const course = availableClasses.find((item) => item.key === opt.value);
                if (course) {
                    opt.textContent = formatOptionLabel(course);
                }
            });
            extraCreditSection.style.display = 'block';
            extraCreditSelect.disabled = false;
            extraCreditSelect.value = key;
            renderExtraCreditSummary();
        }

        async function grantExtraCredit(selectedKey, headers) {
            if (!validatedStudent) {
                extraCreditStatus.textContent = 'Validate your student ID first.';
                extraCreditStatus.className = statusClasses.error;
                return;
            }
            if (extraCreditClaimed) {
                extraCreditStatus.textContent = 'Extra credit already claimed.';
                extraCreditStatus.className = statusClasses.error;
                return;
            }
            if (!selectedKey) {
                extraCreditStatus.textContent = 'Pick a class before claiming extra credit.';
                extraCreditStatus.className = statusClasses.error;
                return;
            }
            const pointsToAdd = 1;

            extraCreditStatus.textContent = 'Adding +1 extra credit...';
            extraCreditStatus.className = statusClasses.muted;

            try {
                const response = await fetch('/extra-credit', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({ student_id: validatedStudent.student_id, class_key: selectedKey, points: pointsToAdd })
                });
                const payload = await response.json();
                if (!response.ok) {
                    const detail = payload.detail || payload.message || 'Could not update extra credit.';
                    throw new Error(detail);
                }
                updateClassOptionLabel(selectedKey, payload.total_points);
                extraCreditClaimed = true;
                extraCreditStatus.textContent = `${payload.class_name} now has ${payload.total_points} extra credit point${payload.total_points === 1 ? '' : 's'}.`;
                extraCreditStatus.className = statusClasses.ok;
                extraCreditSelect.disabled = true;
                extraCreditSection.style.opacity = '0.7';
                return payload;
            } catch (error) {
                extraCreditStatus.textContent = error.message || 'Could not update extra credit.';
                extraCreditStatus.className = statusClasses.error;
                throw error;
            }
        }

        function resetForm() {
            validatedStudent = null;
            studentIdInput.value = '';
            studentText.value = '';
            studentText.disabled = true;
            executeButton.disabled = true;
            setStatus('Enter your student ID to continue.', 'muted');
            resetExtraCredit();
            resultPanel.style.display = 'none';
            resourceLink.style.display = 'none';
            updateWordCounter();
        }

        function togglePanel(show) {
            formPanel.classList.toggle('active', show);
            tryButton.disabled = show;
            if (show) {
                studentIdInput.focus();
            } else {
                resetForm();
                apiKeyInput.value = '';
            }
        }

        studentIdInput.addEventListener('input', () => {
            validatedStudent = null;
            studentText.disabled = true;
            executeButton.disabled = true;
            resetExtraCredit();
            setStatus('Re-validate your student ID to continue.', 'muted');
            updateWordCounter();
        });

        validateButton.addEventListener('click', async () => {
            const studentId = studentIdInput.value.trim();
            if (!studentId) {
                alert('Please enter your student ID.');
                return;
            }
            validateButton.disabled = true;
            setStatus('Checking ID...', 'muted');

            const headers = { 'Content-Type': 'application/json' };
            const apiKey = apiKeyInput.value.trim();
            if (apiKey) {
                headers['X-API-Key'] = apiKey;
            }

            try {
                const response = await fetch('/validate-student', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({ student_id: studentId })
                });
                const payload = await response.json();
                if (!response.ok || !payload.valid) {
                    const detail = payload.detail || payload.message || 'Student ID not recognized.';
                    throw new Error(detail);
                }
                validatedStudent = payload;
                extraCreditClaimed = !!payload.has_extra;
                populateExtraCreditOptions(payload.classes || []);
                if (extraCreditClaimed) {
                    extraCreditSelect.disabled = true;
                    extraCreditSection.style.opacity = '0.7';
                    extraCreditStatus.textContent = 'Extra credit already claimed.';
                    extraCreditStatus.className = statusClasses.error;
                }
                studentText.disabled = false;
                updateExecuteState();
                const displayName = [payload.first_name, payload.last_name].filter(Boolean).join(' ').trim();
                const nameText = displayName ? ` for ${displayName}` : '';
                setStatus(`ID verified${nameText}. You can enter your response.`, 'ok');
                studentText.focus();
            } catch (error) {
                validatedStudent = null;
                studentText.disabled = true;
                executeButton.disabled = true;
                resetExtraCredit();
                setStatus(error.message || 'Student ID not recognized.', 'error');
                updateWordCounter();
            } finally {
                validateButton.disabled = false;
            }
        });

        studentText.addEventListener('input', () => {
            updateExecuteState();
        });
        consentCheckbox.addEventListener('change', () => {
            updateExecuteState();
        });

        tryButton.addEventListener('click', () => togglePanel(true));
        cancelButton.addEventListener('click', () => togglePanel(false));

        executeButton.addEventListener('click', async () => {
            if (!validatedStudent) {
                alert('Please validate your student ID before submitting.');
                return;
            }
            const text = studentText.value.trim();
            if (!text) {
                alert('Please enter student text before executing.');
                return;
            }
            executeButton.disabled = true;
            executeButton.textContent = 'Sending...';
            resultPanel.style.display = 'block';
            resultBody.textContent = 'Waiting for response...';

            const headers = { 'Content-Type': 'application/json' };
            const apiKey = apiKeyInput.value.trim();
            if (apiKey) {
                headers['X-API-Key'] = apiKey;
            }

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({ text, student_id: validatedStudent.student_id, consent: true })
                });
            const payload = await response.json();
            if (!response.ok) {
                const detail = payload.detail || payload.message || 'Request failed.';
                throw new Error(detail);
            }
            resultBody.textContent = 'Thanks. Your submission was received and will be reviewed.';
            if (availableClasses.length) {
                const chosenClassKey = extraCreditSelect.value || (availableClasses[0] ? availableClasses[0].key : '');
                try {
                    const creditResult = await grantExtraCredit(chosenClassKey, headers);
                    if (creditResult && creditResult.class_name) {
                        const awarded = creditResult.points_awarded || 1;
                        resultBody.textContent = `Thanks. Your submission was received. +${awarded} extra credit added to ${creditResult.class_name}.`;
                        extraCreditClaimed = true;
                    }
                } catch (err) {
                    // Already reported in the extra credit status UI
                }
            } else {
                    extraCreditStatus.textContent = 'No class selected for extra credit.';
                    extraCreditStatus.className = statusClasses.muted;
                }
            } catch (error) {
                resultBody.textContent = 'Request failed: ' + error;
            } finally {
                executeButton.disabled = false;
                executeButton.textContent = 'Execute';
                resourceLink.style.display = 'block';
            }
        });
    })();
    </script>
</body>
</html>
"""
CUSTOM_DOCS_HTML = CUSTOM_DOCS_HTML.replace("__MIN_WORDS__", str(MIN_RESPONSE_WORDS))

STRESS_MODEL_CANDIDATE = os.getenv("HACKATHON_STRESS_MODEL", "models/stress-model")
STRESS_BASE_MODEL = os.getenv("HACKATHON_STRESS_BASE", "roberta-large")
STRESS_TOKENIZER_FALLBACK = os.getenv(
    "HACKATHON_STRESS_TOKENIZER", STRESS_BASE_MODEL
)
MENTAL_MODEL_NAME = os.getenv(
    "HACKATHON_MENTAL_MODEL", "dsuram/distilbert-mentalhealth-classifier"
)
EMOTION_MODEL_NAME = os.getenv(
    "HACKATHON_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
)

MENTAL_SUICIDAL_KEYWORDS = [
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "wish i was dead",
    "wish i were dead",
    "take my life",
    "never wake up",
]
MENTAL_SUICIDAL_SEMANTIC_PATTERNS = [
    "don't trust myself at night",
    "don't trust myself overnight",
    "sleep and not wake",
    "never waking up",
    "hospitalized so i can rest",
    "get hospitalized so i can rest",
    "be gone and not come back",
    "stepping into traffic",
    "walk into traffic",
    "walk in front of a car",
    "avoid crossing streets",
    "intrusive picture of",
    "don't trust myself near cars",
    "don't trust myself near bridges",
]

ACADEMIC_STRESS_SCALE = {
    "0": "None",
    "1": "Very low",
    "2": "Mild",
    "3": "Moderate",
    "4": "High",
    "5": "Severe",
}

THERAPIST_CREDENTIALS_PATH = Path(
    os.getenv("HACKATHON_THERAPIST_CSV", str(Path("data") / "therapists.csv"))
)
THERAPIST_AUTH_CACHE: Dict[str, Dict[str, str]] = {}
THERAPIST_AUTH_MTIME: float = 0.0
THERAPIST_SESSIONS: Dict[str, str] = {}

MAX_ANALYSIS_HISTORY = int(os.getenv("HACKATHON_HISTORY_LIMIT", "50"))
RECENT_ANALYSES: List[Dict[str, Any]] = []
STUDENT_ROSTER_PATH = Path(
    os.getenv("HACKATHON_STUDENT_ROSTER", str(Path("data") / "student_roster.csv"))
)
REQUIRED_ROSTER_COLUMNS = ("first_name", "last_name", "student_id")
ROSTER_STATUS_COLUMNS = ["has_extra", "hipaa_consent"]
ROSTER_CLASS_COLUMNS = [
    "class_one",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
    "extra_swipe",
]
ROSTER_EXTRA_CREDIT_COLUMNS = [
    "extra_credit_class_1",
    "extra_credit_class_2",
    "extra_credit_class_3",
    "extra_credit_class_4",
    "extra_credit_class_5",
    "extra_credit_class_6",
    "extra_credit_class_7",
    "extra_credit_class_8",
    "extra_credit_extra_swipe",
]
CLASS_TO_EXTRA_CREDIT = dict(zip(ROSTER_CLASS_COLUMNS, ROSTER_EXTRA_CREDIT_COLUMNS))
EXTRA_SWIPE_DEFAULT_NAME = "Add one extra dining swipe"

def record_analysis(
    source: str, text: str, result: Dict[str, Any], student_id: Optional[str] = None
) -> None:
    """Persist a short in-memory history so therapists can review results."""
    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": source,
        "student_id": str(student_id or "N/A"),
        "text": text,
        "academic_stress": result.get("academic_stress") or {},
        "mental_health": result.get("mental_health") or {},
        "emotions": result.get("emotions") or {},
        "mental_health_flags": result.get("mental_health_flags") or {},
        "alert_metadata": result.get("alert_metadata") or {},
        "high_risk": bool(result.get("alert_metadata")),
    }
    RECENT_ANALYSES.append(entry)
    if len(RECENT_ANALYSES) > MAX_ANALYSIS_HISTORY:
        RECENT_ANALYSES.pop(0)

def _format_score_label(title: str, label: Any) -> str:
    """Return a human label, expanding numeric academic stress levels."""
    text = str(label)
    if title.lower().startswith("academic stress"):
        desc = ACADEMIC_STRESS_SCALE.get(text)
        if desc:
            return f"{text} ({desc})"
    return text

def _summarize_academic_stress(scores: Dict[str, Any]) -> Dict[str, float]:
    """Collapse academic stress into two buckets: None vs Stress (1-5)."""
    if not scores:
        return {}
    none_keys = {"0", "none", "no stress"}
    none_total = 0.0
    stress_total = 0.0
    for label, raw_score in scores.items():
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            continue
        normalized = str(label).strip().lower()
        if normalized in none_keys:
            none_total += value
        else:
            stress_total += value
    return {"None": none_total, "Stress": stress_total}

def _score_block_html(title: str, scores: Dict[str, Any]) -> str:
    """Render a short HTML list of the top labels for a score dictionary."""
    if not scores:
        return f"<div class='score-block'><h4>{html.escape(title)}</h4><p>No data</p></div>"
    display_scores = (
        _summarize_academic_stress(scores)
        if title.lower().startswith("academic stress")
        else scores
    )
    ranked = sorted(display_scores.items(), key=lambda item: item[1], reverse=True)
    bullets = "".join(
        f"<li><strong>{html.escape(_format_score_label(title, label))}</strong>: {score:.2f}</li>"
        for label, score in ranked[:5]
    )
    return (
        "<div class='score-block'>"
        f"<h4>{html.escape(title)}</h4>"
        f"<ul>{bullets}</ul>"
        "</div>"
    )

def _snippet(text: str, limit: Optional[int] = None) -> str:
    """Return HTML-safe student text, optionally trimmed."""
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if limit and len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "..."
    return html.escape(cleaned or "No student text supplied.")

def _word_count(text: str) -> int:
    """Count words using basic whitespace splitting."""
    return len([token for token in (text or "").split() if token.strip()])

def _load_therapist_credentials() -> Dict[str, Dict[str, str]]:
    """Load therapist auth records from CSV with a simple mtime cache."""
    global THERAPIST_AUTH_CACHE, THERAPIST_AUTH_MTIME
    try:
        mtime = THERAPIST_CREDENTIALS_PATH.stat().st_mtime
    except FileNotFoundError:
        THERAPIST_AUTH_CACHE = {}
        THERAPIST_AUTH_MTIME = 0.0
        return {}

    if THERAPIST_AUTH_CACHE and mtime == THERAPIST_AUTH_MTIME:
        return THERAPIST_AUTH_CACHE

    records: Dict[str, Dict[str, str]] = {}
    with THERAPIST_CREDENTIALS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fields = {k.lower(): (v or "").strip() for k, v in row.items()}
            username = fields.get("username")
            password = fields.get("password")
            if not username or not password:
                continue
            therapist_id = (
                fields.get("therapist_id")
                or fields.get("therapist id")
                or fields.get("therapist_id_number")
                or fields.get("therapist id number")
                or ""
            )
            name = fields.get("therapist_name") or fields.get("name") or ""
            first_responder = (fields.get("first_responder") or "").lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            record = {
                "username": username,
                "password": password,
                "therapist_id": therapist_id,
                "therapist_name": name,
                "first_responder": first_responder,
            }
            records[username.lower()] = record
            if therapist_id:
                records.setdefault(str(therapist_id).lower(), record)

    THERAPIST_AUTH_CACHE = records
    THERAPIST_AUTH_MTIME = mtime
    return records

def _validate_therapist_credentials(username: str, password: str) -> bool:
    """Check supplied username/password against the therapist CSV."""
    if not username or not password:
        return False
    entries = _load_therapist_credentials()
    match = entries.get(username.strip().lower())
    if not match:
        return False
    return secrets.compare_digest(match.get("password", ""), password.strip())

def _active_therapist_session(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    return THERAPIST_SESSIONS.get(token)

def _get_therapist_record(identifier: str) -> Optional[Dict[str, str]]:
    entries = _load_therapist_credentials()
    return entries.get(str(identifier).strip().lower())

def _issue_therapist_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    THERAPIST_SESSIONS[token] = username
    return token

def _kill_therapist_session(token: Optional[str]) -> None:
    if token and token in THERAPIST_SESSIONS:
        THERAPIST_SESSIONS.pop(token, None)

def _therapist_login_page(message: str = "") -> str:
    hint = (
        "<p style='margin-top:8px;color:#475569;font-size:0.95rem;'>"
        "Use your therapist username/password to proceed. Students must submit at least one response before the dashboard is available.</p>"
    )
    error_html = (
        f"<p style='color:#b91c1c;font-weight:600;'>{html.escape(message)}</p>"
        if message
        else ""
    )
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Therapist Login</title>
        <style>
            body {{ font-family: "Segoe UI", Arial, sans-serif; background:#f4f6fb; padding:60px 16px; }}
            .card {{ max-width:480px; margin:0 auto; background:#fff; padding:28px; border-radius:14px; box-shadow:0 18px 45px rgba(15,23,42,0.08); }}
            label {{ display:block; margin-top:12px; font-weight:600; color:#0f172a; }}
            input {{ width:100%; padding:10px 12px; border:1px solid #cbd5f5; border-radius:10px; font-size:1rem; margin-top:6px; }}
            button {{ margin-top:18px; width:100%; padding:12px; background:#145da0; color:#fff; border:none; border-radius:10px; font-size:1rem; cursor:pointer; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Therapist Login</h2>
            {error_html}
            <form method="post" action="/therapist-login">
                <label for="username">Username</label>
                <input id="username" name="username" type="text" autocomplete="username" required />
                <label for="password">Password</label>
                <input id="password" name="password" type="password" autocomplete="current-password" required />
                <label for="key">Therapist Key (optional)</label>
                <input id="key" name="key" type="text" placeholder="If provided, overrides username/password" />
                <button type="submit">Sign in</button>
            </form>
            {hint}
        </div>
    </body>
    </html>
    """

def _authenticate_therapist(
    request: Request,
    therapist_key: Optional[str],
    credentials: Optional[HTTPBasicCredentials],
) -> (Optional[Dict[str, str]], Optional[str]):
    """Return (therapist_record, session_token_to_set) or (None, None) if unauthenticated."""
    session_token = request.cookies.get("therapist_session")
    session_user = _active_therapist_session(session_token)
    if session_user:
        rec = _get_therapist_record(session_user)
        return rec, None

    if credentials and _validate_therapist_credentials(credentials.username, credentials.password):
        token = _issue_therapist_session(credentials.username)
        rec = _get_therapist_record(credentials.username)
        return rec, token

    return None, None

def _therapist_wait_page() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Awaiting Submission</title>
        <style>
            body { font-family: "Segoe UI", Arial, sans-serif; background:#f4f6fb; padding:60px 16px; }
            .card { max-width:520px; margin:0 auto; background:#fff; padding:28px; border-radius:14px; box-shadow:0 18px 45px rgba(15,23,42,0.08); text-align:center; }
            h2 { margin:0 0 12px; }
            p { color:#475569; }
            .hint { color:#1e293b; font-weight:600; margin-top:14px; }
            .refresh-btn { margin-top:10px; background:#145da0; color:white; border:none; border-radius:8px; padding:10px 16px; cursor:pointer; }
            .refresh-btn:hover { background:#0f4c81; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Waiting for a student submission</h2>
            <p>Once a student submits a response, this page will reload with their analysis.</p>
            <p class="hint">Auto-refreshing every 15 seconds.</p>
            <button class="refresh-btn" onclick="window.location.reload()">Refresh now</button>
        </div>
        <script>
            const REFRESH_MS = 15000;
            setTimeout(() => window.location.reload(), REFRESH_MS);
        </script>
    </body>
    </html>
    """

# Safe Tokenizer Loader (handles missing tokenizer files)
def load_tokenizer(model_name: str, fallback_model_name: Optional[str] = None):

    def _needs_fallback(error: TypeError) -> bool:
        message = str(error)
        return "path should be string" in message and "NoneType" in message

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except TypeError as err:
        if not _needs_fallback(err):
            raise

        candidate = fallback_model_name
        if not candidate:
            config = AutoConfig.from_pretrained(model_name)
            candidate = (
                getattr(config, "_name_or_path", None)
                or getattr(config, "base_model_name_or_path", None)
            )

        if not candidate:
            raise

        logger.warning(
            f"Tokenizer missing for '{model_name}'. Falling back to '{candidate}'."
        )
        return AutoTokenizer.from_pretrained(candidate)

# Label helper
def _labels_from_model(model, num_scores: int):
    """Return readable labels for a model output vector."""
    return [model.config.id2label.get(i, str(i)) for i in range(num_scores)]

def _resolve_model_source(candidate: Optional[str], fallback: str) -> str:
    """Pick the on-disk fine-tuned model when available, else fall back."""
    if candidate:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate
        if "/" in candidate or candidate.startswith("http"):
            return candidate
        if candidate != fallback:
            logger.warning(
                "Model directory '%s' not found. Falling back to '%s'.",
                candidate,
                fallback,
            )
    return fallback

def _normalize_label_value(value) -> str:
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)

def _parse_label_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    return labels or None

def _read_bool_env(var_name: str) -> Optional[bool]:
    raw = os.getenv(var_name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "Environment variable %s=%s is not a valid boolean flag. Ignoring.",
        var_name,
        raw,
    )
    return None


def _order_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = (
        list(REQUIRED_ROSTER_COLUMNS)
        + ROSTER_STATUS_COLUMNS
        + ROSTER_CLASS_COLUMNS
        + ROSTER_EXTRA_CREDIT_COLUMNS
    )
    existing_preferred = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_preferred]
    return df[existing_preferred + remaining]


def _parse_extra_credit(raw_value: Any) -> int:
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return 0


def _parse_bool_flag(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _load_roster_dataframe() -> pd.DataFrame:
    if not STUDENT_ROSTER_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Student roster file not found at {STUDENT_ROSTER_PATH}.",
        )
    try:
        df = pd.read_csv(STUDENT_ROSTER_PATH, dtype=str)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise HTTPException(
            status_code=500,
            detail=f"Could not read student roster: {exc}",
        ) from exc

    df = df.rename(columns=lambda col: str(col).strip().lower())
    # Backward compatibility: migrate legacy column name if present
    if "has_extra" not in df.columns and "has_extra_credit" in df.columns:
        df["has_extra"] = df["has_extra_credit"]

    missing = [col for col in REQUIRED_ROSTER_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Student roster missing required columns: {', '.join(missing)}.",
        )

    for col in ROSTER_CLASS_COLUMNS + ROSTER_EXTRA_CREDIT_COLUMNS + ROSTER_STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = _order_roster_columns(df)
    # Normalize has_extra as boolean-like strings for consistency
    df["has_extra"] = df["has_extra"].apply(_parse_bool_flag)
    df["hipaa_consent"] = df["hipaa_consent"].apply(_parse_bool_flag)
    return df


def _extract_class_entries(row: pd.Series) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    for idx, class_col in enumerate(ROSTER_CLASS_COLUMNS):
        name = str(row.get(class_col) or "").strip()
        if class_col == "extra_swipe" and not name:
            name = EXTRA_SWIPE_DEFAULT_NAME
        if not name:
            continue
        extra_col = ROSTER_EXTRA_CREDIT_COLUMNS[idx]
        name_lower = name.lower()
        if name_lower in {"nan", "none"}:
            continue
        points = _parse_extra_credit(row.get(extra_col))
        classes.append(
            {
                "key": class_col,
                "name": name,
                "extra_credit_column": extra_col,
                "extra_credit_points": points,
            }
        )
    return classes


def _total_extra_credit(row: pd.Series) -> int:
    total = 0
    for col in ROSTER_EXTRA_CREDIT_COLUMNS:
        total += _parse_extra_credit(row.get(col))
    return total


def _get_student_row(df: pd.DataFrame, student_id: str) -> pd.Series:
    normalized = str(student_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Student ID is required.")
    mask = df["student_id"].astype(str).str.strip() == normalized
    if not mask.any():
        raise HTTPException(
            status_code=403,
            detail="Student ID not recognized. Please enter a valid ID.",
        )
    idx = df.index[mask][0]
    row = df.loc[idx].copy()
    row["_roster_index"] = idx
    row["has_extra"] = _parse_bool_flag(row.get("has_extra"))
    row["hipaa_consent"] = _parse_bool_flag(row.get("hipaa_consent"))
    return row


def _load_student_roster(include_classes: bool = False) -> Dict[str, Dict[str, Any]]:
    df = _load_roster_dataframe()
    df = df.dropna(subset=["student_id"])
    roster: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        student_id = str(row["student_id"]).strip()
        if not student_id:
            continue
        entry: Dict[str, Any] = {
            "first_name": str(row["first_name"]).strip(),
            "last_name": str(row["last_name"]).strip(),
            "has_extra": _parse_bool_flag(row.get("has_extra")),
            "has_consent": _parse_bool_flag(row.get("hipaa_consent")),
        }
        if include_classes:
            entry["classes"] = _extract_class_entries(row)
        roster[student_id] = entry

    if not roster:
        raise HTTPException(
            status_code=500,
            detail="Student roster is empty. Add at least one student row.",
        )
    return roster


def validate_student_id(student_id: str, include_classes: bool = False) -> Dict[str, Any]:
    normalized = str(student_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Student ID is required.")

    roster = _load_student_roster(include_classes=include_classes)
    match = roster.get(normalized)
    if not match:
        raise HTTPException(
            status_code=403,
            detail="Student ID not recognized. Please enter a valid ID.",
        )
    return {"student_id": normalized, **match}


def _increment_extra_credit(student_id: str, class_key: str, points: int = 1) -> Dict[str, Any]:
    df = _load_roster_dataframe()
    row = _get_student_row(df, student_id)
    normalized_key = str(class_key or "").strip().lower()
    if not normalized_key:
        raise HTTPException(status_code=400, detail="Class selection is required.")
    if normalized_key not in CLASS_TO_EXTRA_CREDIT:
        raise HTTPException(status_code=400, detail="Invalid class selection.")
    try:
        points_int = int(points)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Points must be a whole number.")
    if points_int != 1:
        raise HTTPException(status_code=400, detail="Points per request must be exactly 1.")

    current_total = _total_extra_credit(row)
    already_flagged = _parse_bool_flag(row.get("has_extra"))
    if current_total >= 1 or already_flagged:
        raise HTTPException(
            status_code=400,
            detail="Extra credit already claimed for this student.",
        )

    class_name = str(row.get(normalized_key) or "").strip()
    if not class_name:
        raise HTTPException(
            status_code=400,
            detail="Selected class was not found for this student.",
        )

    extra_col = CLASS_TO_EXTRA_CREDIT[normalized_key]
    current_points = _parse_extra_credit(row.get(extra_col))
    new_total = current_points + points_int

    roster_idx = int(row["_roster_index"])
    df.at[roster_idx, extra_col] = new_total
    df.at[roster_idx, "has_extra"] = True
    df = _order_roster_columns(df)
    df.to_csv(STUDENT_ROSTER_PATH, index=False)

    return {
        "student_id": student_id,
        "class_key": normalized_key,
        "class_name": class_name,
        "extra_credit_column": extra_col,
        "points_awarded": points_int,
        "total_points": new_total,
    }


def _set_student_consent(student_id: str, consent: bool = True) -> None:
    df = _load_roster_dataframe()
    row = _get_student_row(df, student_id)
    roster_idx = int(row["_roster_index"])
    df.at[roster_idx, "hipaa_consent"] = bool(consent)
    df = _order_roster_columns(df)
    df.to_csv(STUDENT_ROSTER_PATH, index=False)


ALERT_OUTPUT_DIR = Path(
    os.getenv("HACKATHON_ALERT_DIR", str(Path.cwd() / "alerts"))
)
_alert_env = _read_bool_env("HACKATHON_ENCRYPT_ON_ALERT")
ALERT_ENCRYPTION_ENABLED = True if _alert_env is None else _alert_env
SUICIDAL_FALLBACK_THRESHOLD = float(os.getenv("HACKATHON_SUICIDAL_AUTO_THRESHOLD", "70"))


def _clean_comment_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not any(ch.isalpha() for ch in text):
        return None
    return text


def _get_score_case_insensitive(scores: Dict[str, float], label: str) -> float:
    target = label.lower()
    for key, value in scores.items():
        if key.lower() == target:
            return value
    return 0.0

def _get_label_index_case_insensitive(model, target: str) -> Optional[int]:
    """Return index of label in model.config.id2label matching target (case-insensitive)."""
    target_lower = target.lower()
    for idx, label in model.config.id2label.items():
        if str(label).lower() == target_lower:
            return int(idx)
    return None

def _assign_score_case_insensitive(scores: Dict[str, float], label: str, value: float) -> str:
    """Update a score dict while preserving original label casing if present."""
    target_label = next((key for key in scores.keys() if key.lower() == label.lower()), label)
    scores[target_label] = value
    return target_label

def _segment_text(text: str) -> List[str]:
    """Split text into rough sentence/phrase segments for local risk scanning."""
    if not text:
        return []
    # Split on common sentence/phrase delimiters
    import re
    parts = re.split(r"[\\.;!?\\n]+", text)
    # Keep reasonable-length segments
    return [seg.strip() for seg in parts if len(seg.strip()) >= 20]

def _max_suicidal_over_segments(text: str, label_idx: int) -> Optional[float]:
    """Compute max suicidality percent over text segments to avoid dilution."""
    segments = _segment_text(text)
    if not segments:
        return None
    max_score = None
    for seg in segments:
        inputs = mental_tokenizer(seg, return_tensors="pt", truncation=True)
        outputs = mental_model(**inputs)
        percent = logits_to_percentages(outputs.logits, multi_label=MENTAL_IS_MULTI_LABEL)
        if 0 <= label_idx < len(percent):
            score = float(percent[label_idx])
            max_score = score if max_score is None else max(max_score, score)
    return max_score

def _boost_suicidal_score(
    mental_scores: Dict[str, float], emotion_scores: Dict[str, float], text_lower: str
) -> Tuple[float, List[str], bool, float, List[str]]:
    """
    Apply rule-based boosts to the suicidal score using depression, stress, and sadness.

    Returns: (boosted_score, reasons, mentions_suicide, raw_score, patterns_hit)
    """
    reasons: List[str] = []
    suicidal_base = _get_score_case_insensitive(mental_scores, "suicidal")
    depression_score = _get_score_case_insensitive(mental_scores, "depression")
    stress_score = _get_score_case_insensitive(mental_scores, "stress")
    sadness_score = _get_score_case_insensitive(emotion_scores, "sadness")
    fear_score = _get_score_case_insensitive(emotion_scores, "fear")
    mentions_suicide = any(keyword in text_lower for keyword in MENTAL_SUICIDAL_KEYWORDS)
    patterns_hit = [p for p in MENTAL_SUICIDAL_SEMANTIC_PATTERNS if p in text_lower]

    boosted = suicidal_base
    co_occurs = (
        (depression_score >= 60.0)
        or (stress_score >= 70.0)
        or (sadness_score >= 60.0)
        or (fear_score >= 70.0)
    )

    if patterns_hit and co_occurs:
        boosted = max(boosted, suicidal_base + 10.0)
        reasons.append("Semantic suicidality cues with distress (+10 boost).")
    if depression_score >= 70.0 and sadness_score >= 70.0:
        boosted = max(boosted, suicidal_base + 15.0)
        reasons.append("Depression and sadness both high (+15 boost).")
    if stress_score >= 80.0 and suicidal_base > 5.0:
        boosted = max(boosted, 40.0)
        reasons.append("Stress high and suicidal score above 5% (min clamp to 40).")

    # Cap boosts: preserve raw, cap boosted to 60 while allowing higher raw to pass through
    cap = 60.0
    boosted = max(suicidal_base, min(cap, boosted))
    _assign_score_case_insensitive(mental_scores, "suicidal", boosted)
    return boosted, reasons, mentions_suicide, suicidal_base, patterns_hit


def _should_encrypt_alert() -> bool:
    if not ALERT_ENCRYPTION_ENABLED:
        return False
    if Fernet is None:
        logger.warning(
            "cryptography is not installed; cannot encrypt high-risk submissions."
        )
        return False
    return True


def _encrypt_alert_text(text: str) -> Optional[Dict[str, str]]:
    if not _should_encrypt_alert():
        return None

    key = Fernet.generate_key()
    cipher = Fernet(key).encrypt(text.encode("utf-8"))
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    token = secrets.token_hex(4)
    ALERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"alert_{timestamp}_{token}"
    cipher_path = ALERT_OUTPUT_DIR / f"{base_name}.enc"
    key_path = ALERT_OUTPUT_DIR / f"{base_name}.key"
    cipher_path.write_bytes(cipher)
    key_path.write_bytes(key)
    logger.info("Encrypted high-risk submission to %s (key saved to %s).", cipher_path, key_path)
    return {
        "ciphertext_path": str(cipher_path),
        "key_path": str(key_path),
        "created_at": timestamp,
    }


def _load_csv_dataset(train_file: str, text_column: str, label_column: str):
    df = pd.read_csv(train_file, usecols=[text_column, label_column])
    return df.dropna()


def _load_text_label_dataframe(
    file_path: str, text_column: str, label_column: str
) -> pd.DataFrame:
    suffix = Path(file_path).suffix.lower()
    usecols = [text_column, label_column]
    if suffix == ".csv":
        df = pd.read_csv(file_path, usecols=usecols)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(
            file_path,
            lines=suffix == ".jsonl",
        )
        df = df[usecols]
    else:
        raise ValueError(
            f"Unsupported augment file '{file_path}'. Provide CSV, JSON, or JSONL."
        )
    return df.dropna()


def _append_augmented_samples(
    base_df: pd.DataFrame,
    augment_paths: Sequence[str],
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    frames = [base_df]
    for extra_path in augment_paths:
        extra_df = _load_text_label_dataframe(extra_path, text_column, label_column)
        frames.append(extra_df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna()
    combined = combined.drop_duplicates(subset=[text_column, label_column])
    return combined


def _resolve_studentlife_root(path_str: str) -> Path:
    base = Path(path_str).expanduser()
    candidates = [
        base,
        base / "dataset",
        base / "dataset" / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "EMA" / "response").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate 'EMA/response' inside '{path_str}'. Check the dataset path."
    )


def _build_studentlife_dataframe(dataset_root: str, window_seconds: int = 600) -> pd.DataFrame:
    root = _resolve_studentlife_root(dataset_root)
    stress_dir = root / "EMA" / "response" / "Stress"
    comment_dir = root / "EMA" / "response" / "Comment"
    if not stress_dir.exists() or not comment_dir.exists():
        raise FileNotFoundError(
            f"Missing Stress/Comment folders under '{root}'. Found stress={stress_dir.exists()}, comment={comment_dir.exists()}."
        )

    samples: List[Dict[str, str]] = []
    for stress_path in sorted(stress_dir.glob("Stress_*.json")):
        suffix = stress_path.name.replace("Stress_", "")
        comment_path = comment_dir / f"Comment_{suffix}"
        if not comment_path.exists():
            continue

        stress_data = json.loads(stress_path.read_text(encoding="utf-8"))
        comment_data = json.loads(comment_path.read_text(encoding="utf-8"))
        stress_entries = sorted(
            (
                int(entry["resp_time"]),
                str(entry["level"]),
            )
            for entry in stress_data
            if entry.get("resp_time") is not None and entry.get("level") is not None
        )
        if not stress_entries:
            continue

        stress_times = [t for t, _ in stress_entries]
        stress_levels = [lvl for _, lvl in stress_entries]

        for comment_entry in comment_data:
            text = _clean_comment_text(
                comment_entry.get("comment") or comment_entry.get("null")
            )
            resp_time = comment_entry.get("resp_time")
            if not text or resp_time is None:
                continue
            resp_time = int(resp_time)
            idx = bisect.bisect_left(stress_times, resp_time)
            for offset in (-1, 0, 1):
                j = idx + offset
                if 0 <= j < len(stress_times):
                    candidate_time = stress_times[j]
                    if abs(candidate_time - resp_time) <= window_seconds:
                        samples.append({"text": text, "label": stress_levels[j]})
                        break

    df = pd.DataFrame(samples)
    return df.dropna().drop_duplicates().reset_index(drop=True)

def _detect_dataset_loader(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported dataset format for '{file_path}'. Use CSV or JSON.")

def _load_training_dataset(train_path: str, eval_path: Optional[str]):
    sample = train_path or eval_path
    if not sample:
        raise ValueError("Provide at least --train-file to start fine-tuning.")
    data_format = _detect_dataset_loader(sample)
    data_files: Dict[str, str] = {}
    if train_path:
        data_files["train"] = train_path
    if eval_path:
        data_files["validation"] = eval_path
    return load_dataset(data_format, data_files=data_files)

def _collect_label_list(dataset, label_column: str, provided: Optional[Sequence[str]]):
    if provided:
        return list(provided)
    seen: List[str] = []
    for split in dataset:
        for value in dataset[split][label_column]:
            normalized = _normalize_label_value(value)
            if normalized not in seen:
                seen.append(normalized)
    if not seen:
        raise ValueError(f"No labels found in '{label_column}'.")
    return seen

def _build_preprocess_function(tokenizer, text_column: str, label_column: str, label2id: Dict[str, int], max_length: int):
    def preprocess(batch):
        tokenized = tokenizer(
            batch[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [
            label2id[_normalize_label_value(label)] for label in batch[label_column]
        ]
        return tokenized

    return preprocess

def _compute_accuracy_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float(np.mean(predictions == labels))
    return {"accuracy": accuracy}

def _load_stress_bundle(candidate: str, fallback: str):
    identifier = _resolve_model_source(candidate, fallback)
    try:
        tokenizer = load_tokenizer(
            identifier, fallback_model_name=STRESS_TOKENIZER_FALLBACK
        )
        model = AutoModelForSequenceClassification.from_pretrained(identifier)
        return identifier, tokenizer, model
    except OSError as err:
        if identifier == fallback:
            raise
        logger.warning(
            "Unable to load '%s' (%s). Falling back to '%s'.", identifier, err, fallback
        )
        tokenizer = load_tokenizer(fallback, fallback_model_name=STRESS_TOKENIZER_FALLBACK)
        model = AutoModelForSequenceClassification.from_pretrained(fallback)
        return fallback, tokenizer, model

def train_stress_model(
    train_file: str,
    eval_file: Optional[str],
    text_column: str,
    label_column: str,
    output_dir: str,
    base_model_name: str,
    max_length: int = 256,
    epochs: float = 3.0,
    batch_size: int = 16,
    
    learning_rate: float = 1e-5,
    label_list: Optional[Sequence[str]] = None,
    from_csv: bool = False,
):
    """Fine-tune the academic-stress classifier and save it locally."""
    if from_csv:
        if not train_file:
            raise ValueError("--train-file is required when using CSV mode.")
        train_df = _load_csv_dataset(train_file, text_column, label_column)
        dataset_dict = {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
        }
        if eval_file:
            eval_df = _load_csv_dataset(eval_file, text_column, label_column)
            dataset_dict["validation"] = Dataset.from_pandas(
                eval_df, preserve_index=False
            )
        dataset = DatasetDict(dataset_dict)
    else:
        dataset = _load_training_dataset(train_file, eval_file)
    sample_split = "train" if "train" in dataset else next(iter(dataset.keys()))
    for column in (text_column, label_column):
        if column not in dataset[sample_split].column_names:
            raise ValueError(f"Column '{column}' not found in dataset.")

    labels = _collect_label_list(dataset, label_column, label_list)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    preprocess = _build_preprocess_function(
        tokenizer, text_column, label_column, label2id, max_length
    )
    remove_columns = dataset[sample_split].column_names
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=remove_columns,
    )

    eval_dataset = tokenized_dataset.get("validation")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size, 1),
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps" if eval_dataset is not None else "epoch",
        logging_steps=50,
        eval_steps=200 if eval_dataset is not None else None,
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_accuracy_metrics if eval_dataset is not None else None,
    )

    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    mapping_path = Path(output_dir) / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as fp:
        json.dump({"label2id": label2id, "id2label": id2label}, fp, indent=2)
    logger.info("Fine-tuning complete. Saved model artifacts to %s", output_dir)

# Load the CORRECT models (student-targeted + mental health)

# 1. Student Stress / Burnout model
stress_model_name, stress_tokenizer, stress_model = _load_stress_bundle(
    STRESS_MODEL_CANDIDATE, STRESS_BASE_MODEL
)

# 2. Young-adult mental health classifier (depression/anxiety/loneliness/distress)
mental_model_name = MENTAL_MODEL_NAME
mental_tokenizer = load_tokenizer(mental_model_name)
mental_config = AutoConfig.from_pretrained(mental_model_name)
_mental_override = _read_bool_env("HACKATHON_MENTAL_MULTI_LABEL")
if _mental_override is True:
    mental_config.problem_type = "multi_label_classification"
elif _mental_override is False:
    mental_config.problem_type = None
MENTAL_IS_MULTI_LABEL = (
    getattr(mental_config, "problem_type", None) == "multi_label_classification"
)
mental_model = AutoModelForSequenceClassification.from_pretrained(
    mental_model_name, config=mental_config
)
_mental_threshold_raw = os.getenv("HACKATHON_MENTAL_THRESHOLD", "50.0")
try:
    MENTAL_ACTIVE_THRESHOLD = float(_mental_threshold_raw)
except ValueError:
    logger.warning(
        "Invalid HACKATHON_MENTAL_THRESHOLD=%s. Falling back to 50.0%%.",
        _mental_threshold_raw,
    )
    MENTAL_ACTIVE_THRESHOLD = 50.0
MENTAL_ACTIVE_THRESHOLD = max(0.0, min(100.0, MENTAL_ACTIVE_THRESHOLD))
if MENTAL_IS_MULTI_LABEL:
    logger.info(
        "Mental health model loaded in multi-label mode (threshold %.1f%%).",
        MENTAL_ACTIVE_THRESHOLD,
    )

# 3. Emotion classifier
emotion_model_name = EMOTION_MODEL_NAME
emotion_tokenizer = load_tokenizer(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# Logits -> % conversion
def logits_to_percentages(logits: torch.Tensor, multi_label: bool = False):
    if multi_label:
        probs = torch.sigmoid(logits)[0]
    else:
        probs = F.softmax(logits, dim=1)[0]
    return (probs * 100).tolist()

# MAIN ANALYSIS FUNCTION
def analyze_student_text(text: str):

    # === Academic Stress / Burnout ===
    st_inputs = stress_tokenizer(text, return_tensors="pt", truncation=True)
    st_outputs = stress_model(**st_inputs)
    st_percent = logits_to_percentages(st_outputs.logits)
    stress_labels = _labels_from_model(stress_model, len(st_percent))
    stress_scores = {stress_labels[i]: st_percent[i] for i in range(len(st_percent))}

    # === Mental Health (depression / anxiety / loneliness / distress) ===
    mh_inputs = mental_tokenizer(text, return_tensors="pt", truncation=True)
    mh_outputs = mental_model(**mh_inputs)
    mh_percent = logits_to_percentages(
        mh_outputs.logits, multi_label=MENTAL_IS_MULTI_LABEL
    )
    mental_labels = _labels_from_model(mental_model, len(mh_percent))
    mental_scores = {mental_labels[i]: mh_percent[i] for i in range(len(mh_percent))}
    chunk_max = None
    suicidal_idx = _get_label_index_case_insensitive(mental_model, "suicidal")
    if suicidal_idx is not None:
        chunk_max = _max_suicidal_over_segments(text, suicidal_idx)
        if chunk_max is not None:
            suicidal_label = next(
                (key for key in mental_scores if key.lower() == "suicidal"),
                mental_labels[suicidal_idx],
            )
            mental_scores[suicidal_label] = max(
                mental_scores.get(suicidal_label, 0.0), chunk_max
            )

    # === Emotion Model ===
    em_inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    em_outputs = emotion_model(**em_inputs)
    em_percent = logits_to_percentages(em_outputs.logits)
    emotion_labels = _labels_from_model(emotion_model, len(em_percent))
    emotion_scores = {emotion_labels[i]: em_percent[i] for i in range(len(em_percent))}

    # === Suicidal risk boost (rule-conditional) ===
    text_lower = text.lower()
    suicidal_score, suicidal_reasons, mentions_suicide, suicidal_raw, patterns_hit = (
        _boost_suicidal_score(mental_scores, emotion_scores, text_lower)
    )

    # === Mental Health Flags & alerts ===
    mental_flags: Dict[str, Any] = {}
    alert_metadata: Optional[Dict[str, str]] = None
    if MENTAL_IS_MULTI_LABEL:
        mental_flags = {
            label: score
            for label, score in mental_scores.items()
            if score >= MENTAL_ACTIVE_THRESHOLD
        }
        depression_score = _get_score_case_insensitive(mental_scores, "depression")
        suicidal_label = next(
            (key for key in mental_scores if key.lower() == "suicidal"),
            "suicidal",
        )
        if suicidal_score >= MENTAL_ACTIVE_THRESHOLD:
            mental_flags[suicidal_label] = suicidal_score
        elif depression_score >= SUICIDAL_FALLBACK_THRESHOLD and mentions_suicide:
            mental_flags[suicidal_label] = max(
                suicidal_score, depression_score, MENTAL_ACTIVE_THRESHOLD
            )

        suicidal_flagged = any(label.lower() == "suicidal" for label in mental_flags)
        if suicidal_flagged:
            alert_metadata = _encrypt_alert_text(text)
            if alert_metadata is not None:
                alert_metadata["suicidal_raw"] = f"{round(suicidal_raw, 2)}"
                alert_metadata["suicidal_boosted"] = f"{round(suicidal_score, 2)}"
                if chunk_max is not None:
                    alert_metadata["suicidal_chunk_max"] = f"{round(chunk_max, 2)}"
                if suicidal_reasons:
                    alert_metadata["suicidal_boost_reasons"] = "; ".join(suicidal_reasons)

    response = {
        "academic_stress": stress_scores,
        "mental_health": mental_scores,
        "emotions": emotion_scores,
    }
    if MENTAL_IS_MULTI_LABEL:
        response["mental_health_flags"] = mental_flags
    if alert_metadata:
        response["alert_metadata"] = alert_metadata
    return response

# CLI DISPLAY
def display_scores(result: dict):
    """Pretty-print the model outputs so end users can read them."""
    for section, scores in result.items():
        if section == "alert_metadata":
            continue
        if not isinstance(scores, dict):
            continue
        section_name = section.replace("_", " ").title()
        if section == "mental_health_flags" and MENTAL_IS_MULTI_LABEL:
            section_name = (
                f"Mental Health Flags (>= {MENTAL_ACTIVE_THRESHOLD:.1f}%)"
            )
        print(f"\n{section_name}:")
        if not scores:
            print("  - None")
            continue
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for label, score in sorted_scores:
            label_text = label.replace("_", " ").title()
            highlight = (
                section == "mental_health"
                and MENTAL_IS_MULTI_LABEL
                and score >= MENTAL_ACTIVE_THRESHOLD
            )
            prefix = "*" if highlight else "-"
            print(f"  {prefix} {label_text}: {score:.1f}%")
        if section == "mental_health" and MENTAL_IS_MULTI_LABEL:
            print(
                f"  (* indicates labels above {MENTAL_ACTIVE_THRESHOLD:.1f}% multi-label threshold)"
            )
            overlapping = [
                label.replace("_", " ").title()
                for label, score in sorted_scores
                if score >= MENTAL_ACTIVE_THRESHOLD
            ]
            if len(overlapping) > 1:
                print(f"  Overlap detected: {', '.join(overlapping)}")

# API MODELS
class AnalyzeRequest(BaseModel):
    student_id: str = Field(
        ...,
        description="Student ID used to confirm the student is on the approved roster.",
        example="900123456",
    )
    text: str = Field(
        ...,
        description=(
            "Type or paste the student's full response here. "
            "For example: 'I failed my exams and feel hopeless about the semester.'"
        ),
        example="I failed three classes and feel hopeless. I need help.",
    )
    consent: bool = Field(
        ...,
        description="HIPAA consent acknowledgment required to submit.",
        example=True,
    )

class AnalyzeResponse(BaseModel):
    academic_stress: Dict[str, float]
    mental_health: Dict[str, float]
    emotions: Dict[str, float]
    mental_health_flags: Optional[Dict[str, float]] = None
    alert_metadata: Optional[Dict[str, str]] = None


class StudentValidationRequest(BaseModel):
    student_id: str = Field(
        ...,
        description="Student ID to validate before any analysis is allowed.",
        example="900123456",
    )


class StudentClassInfo(BaseModel):
    key: str = Field(
        ...,
        description="Roster column for the selected class (class_one, class_2, ...).",
        example="class_2",
    )
    name: str = Field(..., description="Human-readable class name.", example="Data Mining")
    extra_credit_column: str = Field(
        ...,
        description="Mapped extra credit column that will be incremented.",
        example="extra_credit_class_2",
    )
    extra_credit_points: int = Field(
        0,
        description="Current extra credit total for this class.",
        example=3,
    )


class StudentValidationResponse(BaseModel):
    valid: bool
    student_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    classes: List[StudentClassInfo] = Field(default_factory=list)
    has_extra: bool = False
    has_consent: bool = False
    message: str


class ExtraCreditRequest(BaseModel):
    student_id: str = Field(
        ...,
        description="Validated student ID that owns the class entry.",
        example="900123456",
    )
    class_key: str = Field(
        ...,
        description="Roster class column to increment (class_one through class_8).",
        example="class_one",
    )
    points: int = Field(
        1,
        description="Number of extra credit points to add (must be 1).",
        example=1,
    )


class ExtraCreditResponse(BaseModel):
    student_id: str
    class_key: str
    class_name: str
    extra_credit_column: str
    points_awarded: int
    has_extra: bool
    total_points: int
    message: str

# SECURITY: optional API key
def require_api_key(api_key: Optional[str] = Depends(api_key_header)):
    expected = os.getenv("HACKATHON_API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key

def require_therapist_key(
    request: Request,
    therapist_key: Optional[str] = Depends(therapist_key_header),
    credentials: Optional[HTTPBasicCredentials] = Depends(therapist_basic),
):
    """Gate access to therapist endpoints using either an API key or CSV-backed login."""
    user, _ = _authenticate_therapist(request, therapist_key, credentials)
    if user:
        return True
    raise HTTPException(
        status_code=401,
        detail="Therapist credentials missing or invalid.",
        headers={"WWW-Authenticate": 'Basic realm="Therapist Dashboard"'},
    )


@app.post("/validate-student", response_model=StudentValidationResponse)
def validate_student_endpoint(
    payload: StudentValidationRequest, _: Optional[str] = Depends(require_api_key)
):
    student = validate_student_id(payload.student_id, include_classes=True)
    return {
        "valid": True,
        "student_id": student["student_id"],
        "first_name": student.get("first_name") or None,
        "last_name": student.get("last_name") or None,
        "classes": student.get("classes") or [],
        "has_extra": bool(student.get("has_extra")),
        "has_consent": bool(student.get("has_consent")),
        "message": "Student ID verified.",
    }


@app.get("/docs", include_in_schema=False)
def custom_docs():
    return HTMLResponse(CUSTOM_DOCS_HTML)

@app.get("/therapist", include_in_schema=False, response_class=HTMLResponse)
def therapist_dashboard(
    request: Request,
    therapist_key: Optional[str] = Depends(therapist_key_header),
    credentials: Optional[HTTPBasicCredentials] = Depends(therapist_basic),
):
    user_record, session_token = _authenticate_therapist(request, therapist_key, credentials)
    if not user_record:
        return RedirectResponse(
            url="/therapist-login",
            status_code=303,
        )

    is_first_responder = bool(user_record.get("first_responder"))

    if not RECENT_ANALYSES:
        return HTMLResponse(_therapist_wait_page(), status_code=503)

    view_entries = [e for e in RECENT_ANALYSES if not is_first_responder or e.get("alert_metadata")]

    if not view_entries:
        history_html = "<p class='empty-state'>No submissions have been recorded yet.</p>"
    else:
        cards: List[str] = []
        for entry in reversed(view_entries):
            snippet = _snippet(entry["text"])
            student_id_display = html.escape(str(entry.get("student_id", "N/A")))
            score_blocks = "".join(
                [
                    _score_block_html("Academic Stress", entry["academic_stress"]),
                    _score_block_html("Mental Health", entry["mental_health"]),
                    _score_block_html("Emotions", entry["emotions"]),
                ]
            )
            flags = entry.get("mental_health_flags") or {}
            flag_items = (
                "<li>No high-risk indicators detected.</li>"
                if not flags
                else "".join(
                    f"<li><strong>{html.escape(label)}</strong>: {value:.2f}</li>"
                    if isinstance(value, (int, float))
                    else f"<li><strong>{html.escape(label)}</strong>: {html.escape(str(value))}</li>"
                    for label, value in flags.items()
                )
            )
            alert_box = ""
            if entry.get("alert_metadata"):
                def _to_float(val):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None

                suicidal_pct = _get_score_case_insensitive(
                    entry.get("mental_health") or {}, "suicidal"
                )
                pct_text = f"{suicidal_pct:.2f}%" if suicidal_pct else "N/A"
                raw_val_raw = entry["alert_metadata"].get("suicidal_raw")
                boosted_val_raw = entry["alert_metadata"].get("suicidal_boosted")
                raw_val = _to_float(raw_val_raw)
                boosted_val = _to_float(boosted_val_raw)
                raw_text = f"{raw_val:.2f}%" if isinstance(raw_val, (int, float)) else "N/A"
                boosted_text = f"{boosted_val:.2f}%" if isinstance(boosted_val, (int, float)) else pct_text
                reasons_text = entry["alert_metadata"].get("suicidal_boost_reasons")
                reasons_html = (
                    f"<p class='alert-reasons'>{html.escape(str(reasons_text))}</p>"
                    if reasons_text
                    else ""
                )
                boosted_flag = (
                    isinstance(raw_val, (int, float))
                    and isinstance(boosted_val, (int, float))
                    and boosted_val > raw_val + 0.01
                )
                low_unboosted = (
                    (not boosted_flag)
                    and isinstance(boosted_val, (int, float))
                    and boosted_val < 5.0
                )
                status_lines = []
                if boosted_flag:
                    status_lines.append("Urgent: Contacted Emergency Services")
                    status_lines.append("Counselor Sent Notification")
                elif low_unboosted:
                    status_lines.append("Urgent: Student in Potential Danger")
                    status_lines.append("Counselor Sent Notification")
                else:
                    status_lines.append("Counselor Sent Notification")
                status_html = "".join(f"<p class='alert-status'>{html.escape(line)}</p>" for line in status_lines)
                alert_box = (
                    f"<div class='alert-box urgent'>"
                    "<h4>High-Risk Alert</h4>"
                    f"{status_html}"
                    f"<p class='alert-detail'>Suicidality (raw → boosted): {raw_text} → {boosted_text}</p>"
                    f"{reasons_html}"
                    "</div>"
                )

            cards.append(
                f"""
                <article class="analysis-card">
                    <header>
                        <div>
                            <p class="timestamp">{html.escape(entry["timestamp"])}</p>
                            <p class="source-tag">{html.escape(entry["source"])}</p>
                            <p class="student-id">Student ID: {student_id_display}</p>
                        </div>
                    </header>
                    <p class="student-text">{snippet}</p>
                    <div class="score-grid">
                        {score_blocks}
                        <div class="score-block">
                            <h4>Mental Health Flags</h4>
                            <ul>{flag_items}</ul>
                        </div>
                    </div>
                    {alert_box}
                </article>
                """
            )
        history_html = "".join(cards)

    stress_scale_note = (
        "Academic Stress display: None (level 0) vs Stress (levels 1-5 combined)."
    )

    heading = "First Responder Review Board" if is_first_responder else "Therapist Review Board"
    subtext = (
        f"Campus Police: {FIRST_RESPONDER_CONTACT} — High-risk alerts only."
        if is_first_responder
        else "Secure window for licensed staff to review recent student submissions. Share the /therapist URL with the key only."
    )

    page_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Therapist Dashboard</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: "Segoe UI", Arial, sans-serif;
                background:#f4f6fb;
                margin:0;
                padding:40px 12px 80px;
                color:#0f172a;
            }}
            .refresh-bar {{
                position:sticky;
                top:0;
                z-index:5;
                background:rgba(255,255,255,0.92);
                backdrop-filter: blur(4px);
                border-bottom:1px solid #e2e8f0;
                padding:10px 16px;
                display:flex;
                align-items:center;
                gap:10px;
                justify-content:center;
                font-weight:600;
                color:#0f172a;
            }}
            .refresh-bar button {{
                background:#145da0;
                color:white;
                border:none;
                border-radius:8px;
                padding:8px 14px;
                cursor:pointer;
                font-weight:600;
            }}
            .refresh-bar button:hover {{ background:#0f4c81; }}
            header {{
                max-width:1080px;
                margin:0 auto 24px;
                text-align:center;
            }}
            header h1 {{ margin-bottom:6px; }}
            header p {{ color:#475569; margin:0; }}
            .analysis-card {{
                max-width:1080px;
                margin:0 auto 20px;
                background:#fff;
                padding:24px;
                border-radius:18px;
                box-shadow:0 18px 45px rgba(15,23,42,0.08);
            }}
            .score-grid {{
                display:grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap:16px;
                margin-top:16px;
            }}
            .score-block {{
                background:#f8fafc;
                border-radius:12px;
                padding:12px 16px;
            }}
            .score-block h4 {{
                margin:0 0 8px;
                font-size:1rem;
                color:#0f172a;
            }}
            .score-block ul {{
                margin:0;
                padding-left:20px;
                color:#0f172a;
            }}
            .student-text {{
                font-size:1.05rem;
                margin:12px 0 4px;
                color:#111827;
                white-space: pre-wrap;
                word-break: break-word;
            }}
            .timestamp {{
                font-weight:600;
            }}
            .source-tag {{
                text-transform:uppercase;
                letter-spacing:0.08em;
                font-size:0.8rem;
                color:#475569;
            }}
            .student-id {{
                margin:2px 0 0;
                color:#0f172a;
                font-weight:600;
            }}
            .alert-box {{
                background:#fef3c7;
                border-left:6px solid #f97316;
                padding:12px 16px;
                border-radius:10px;
                margin-top:18px;
            }}
            .alert-box.urgent {{
                background:#fee2e2;
                border-left-color:#dc2626;
            }}
            .alert-box h4 {{
                margin:0 0 8px;
                color:#b45309;
            }}
            .alert-box ul {{
                margin:0;
                padding-left:20px;
            }}
            .alert-status {{
                margin:0 0 6px;
                font-weight:700;
                color:#b91c1c;
            }}
            .alert-detail {{
                margin:0 0 6px;
            }}
            .empty-state {{
                text-align:center;
                margin-top:60px;
                color:#475569;
                font-size:1.1rem;
            }}
            .scale-note {{
                margin-top:8px;
                color:#1e293b;
                font-size:0.95rem;
            }}
        </style>
    </head>
    <body>
        <div class="refresh-bar">
            <span>Auto-refreshing every 15 seconds.</span>
            <button type="button" onclick="window.location.reload()">Refresh now</button>
        </div>
        <header>
            <h1>{html.escape(heading)}</h1>
            <p>{html.escape(subtext)}</p>
            <p class="scale-note">{html.escape(stress_scale_note)}</p>
        </header>
        {history_html}
        <script>
            const REFRESH_MS = 15000;
            setTimeout(() => window.location.reload(), REFRESH_MS);
        </script>
    </body>
    </html>
    """
    response = HTMLResponse(page_html)
    if session_token:
        response.set_cookie(
            "therapist_session",
            session_token,
            httponly=True,
            samesite="Lax",
            secure=False,
        )
    return response


@app.get("/therapist-login", include_in_schema=False, response_class=HTMLResponse)
def therapist_login_page(message: Optional[str] = None):
    return HTMLResponse(_therapist_login_page(message or ""))


@app.post("/therapist-login", include_in_schema=False)
async def therapist_login(
    username: str = Form(""),
    password: str = Form(""),
):
    session_token = None
    authed = False

    if _validate_therapist_credentials(username.strip(), password.strip()):
        authed = True
        session_token = _issue_therapist_session(username.strip())

    if not authed:
        return HTMLResponse(
            _therapist_login_page("Invalid credentials. Please try again."),
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Therapist Dashboard"'},
        )

    response = RedirectResponse(url="/therapist", status_code=303)
    if session_token:
        response.set_cookie(
            "therapist_session",
            session_token,
            httponly=True,
            samesite="Lax",
            secure=False,
        )
    return response


@app.post("/therapist-logout", include_in_schema=False)
async def therapist_logout(request: Request):
    token = request.cookies.get("therapist_session")
    _kill_therapist_session(token)
    response = RedirectResponse(url="/therapist-login", status_code=303)
    response.delete_cookie("therapist_session")
    return response

# API ROUTES
@app.get("/", include_in_schema=False)
def healthcheck():
    return {
        "status": "ok",
        "message": "Use the POST /analyze endpoint (see /docs) and type the student's text into the box labeled 'text'.",
        "steps": [
            "Open /docs in a browser.",
            "Expand POST /analyze, click 'Try it out', and type or paste the response into the text field.",
            "Press Execute to see the scores and any alerts.",
        ],
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(payload: AnalyzeRequest, _: Optional[str] = Depends(require_api_key)):
    validate_student_id(payload.student_id)
    cleaned = payload.text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if not payload.consent:
        raise HTTPException(status_code=400, detail="HIPAA consent must be acknowledged before submission.")
    word_count = _word_count(cleaned)
    if word_count < MIN_RESPONSE_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text must be at least {MIN_RESPONSE_WORDS} words (received {word_count}).",
        )
    try:
        _set_student_consent(payload.student_id, True)
    except HTTPException:
        # If roster write fails for some reason, still proceed with analysis
        pass
    result = analyze_student_text(cleaned)
    record_analysis("api", cleaned, result, student_id=payload.student_id)
    return result


@app.post("/extra-credit", response_model=ExtraCreditResponse)
def extra_credit_endpoint(
    payload: ExtraCreditRequest, _: Optional[str] = Depends(require_api_key)
):
    validate_student_id(payload.student_id)
    credit = _increment_extra_credit(payload.student_id, payload.class_key, payload.points)
    credit["message"] = (
        f"Added +{credit['points_awarded']} extra credit to {credit['class_name']}. "
        f"New total: {credit['total_points']}."
    )
    credit["has_extra"] = True
    return credit

@app.get("/submit", include_in_schema=False, response_class=HTMLResponse)
def show_submission_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Submit Student Text</title>
        <style>
            body {font-family: Arial, sans-serif; background:#f7f7f7; padding:40px;}
            main {max-width: 720px; margin: 0 auto; background:white; padding:30px; border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.08);}
            label {display:block; margin-top:12px; font-weight:600;}
            input[type="text"] {width:100%; padding:10px 12px; margin-top:6px; border:1px solid #cbd5f5; border-radius:8px; font-size:16px;}
            .id-row {display:flex; gap:10px; flex-wrap:wrap; align-items:center;}
            .id-row input {flex:1 1 240px;}
            textarea {width:100%; min-height:180px; padding:12px; font-size:16px;}
            textarea:disabled {background:#e2e8f0; cursor:not-allowed;}
            button {margin-top:16px; font-size:18px; padding:12px 20px; background:#145DA0; color:white; border:none; border-radius:8px; cursor:pointer;}
            button:hover:not(:disabled) {background:#0f4c81;}
            button:disabled {opacity:0.65; cursor:default;}
            .secondary {background:#0ea5e9;}
            .hint {color:#555; margin-top:12px;}
            .status {margin:8px 0 4px; color:#475569;}
            .status.ok {color:#15803d;}
            .status.error {color:#b91c1c;}
        </style>
    </head>
    <body data-min-words="__MIN_WORDS__">
        <main>
            <h1>Analyze a Student Response</h1>
            <form method="post" id="submitForm" novalidate>
                <label for="student_id">Student ID</label>
                <div class="id-row">
                    <input id="student_id" name="student_id" type="text" required placeholder="Enter your student ID" />
                    <button type="button" class="secondary" id="validateSubmitId">Validate ID</button>
                </div>
                <p class="status" id="submitStatus">Validate your student ID to unlock the response box.</p>
                <label for="text">Paste or type the message:</label>
                <textarea id="text" name="text" placeholder="I feel overwhelmed by exams and can't sleep..." disabled></textarea>
                <p class="status muted" style="margin:6px 0 2px;">
                    <label style="display:flex; align-items:center; gap:8px;">
                        <input id="submitConsent" type="checkbox" />
                        <span>I consent to the HIPAA notice.</span>
                    </label>
                </p>
                <p id="submitWordCounter" class="status muted">0/__MIN_WORDS__ words (min).</p>
                <label for="apiKeyInput">Optional API key (X-API-Key header)</label>
                <input id="apiKeyInput" type="text" placeholder="Only needed if the server requires it." />
                <button type="submit" id="submitButton" disabled>Submit for Analysis</button>
            </form>
            <p class="hint">After you submit, watch the PowerShell window running this server. The full analysis is written there.</p>
        </main>
        <script>
        (function(){
            const form = document.getElementById('submitForm');
            const studentId = document.getElementById('student_id');
            const textArea = document.getElementById('text');
            const status = document.getElementById('submitStatus');
            const wordCounter = document.getElementById('submitWordCounter');
            const validateBtn = document.getElementById('validateSubmitId');
            const submitBtn = document.getElementById('submitButton');
            const apiKeyInput = document.getElementById('apiKeyInput');
            const consentBox = document.getElementById('submitConsent');
            const MIN_WORDS = parseInt(document.body.dataset.minWords || "50", 10) || 50;
            let validated = false;

            function setStatus(message, type){
                status.textContent = message;
                status.className = 'status ' + (type || '');
            }

            function countWords(text){
                if (!text) return 0;
                return text.trim().split(/\\s+/).filter(Boolean).length;
            }

            function updateWordCounter(){
                const count = countWords(textArea.value);
                wordCounter.textContent = `${count}/${MIN_WORDS} words (min).`;
                wordCounter.className = 'status ' + (count >= MIN_WORDS ? 'ok' : 'error');
                return count;
            }

            function updateSubmitState(){
                const count = updateWordCounter();
                const hasText = textArea.value.trim().length > 0;
                submitBtn.disabled = !(validated && hasText && count >= MIN_WORDS && consentBox.checked);
            }

            function lockForm(){
                validated = false;
                textArea.value = '';
                textArea.disabled = true;
                submitBtn.disabled = true;
                setStatus('Validate your student ID to unlock the response box.', 'muted');
                updateWordCounter();
            }

            studentId.addEventListener('input', lockForm);

            async function validateId() {
                const id = studentId.value.trim();
                if (!id) {
                    setStatus('Enter your student ID to continue.', 'error');
                    return false;
                }
                validateBtn.disabled = true;
                setStatus('Checking ID...', 'muted');

                const headers = { 'Content-Type': 'application/json' };
                const apiKey = apiKeyInput.value.trim();
                if (apiKey) {
                    headers['X-API-Key'] = apiKey;
                }

                try {
                    const response = await fetch('/validate-student', {
                        method: 'POST',
                        headers,
                        body: JSON.stringify({ student_id: id })
                    });
                    const payload = await response.json();
                    if (!response.ok || !payload.valid) {
                        const detail = payload.detail || payload.message || 'Student ID not recognized.';
                        throw new Error(detail);
                    }
                    validated = true;
                    textArea.disabled = false;
                    setStatus('ID verified. You can enter your response now.', 'ok');
                    updateSubmitState();
                    return true;
                } catch (error) {
                    validated = false;
                    textArea.disabled = true;
                    submitBtn.disabled = true;
                    setStatus(error.message || 'Student ID not recognized.', 'error');
                    return false;
                } finally {
                    validateBtn.disabled = false;
                }
            }

            validateBtn.addEventListener('click', validateId);

            textArea.addEventListener('input', updateSubmitState);
            consentBox.addEventListener('change', updateSubmitState);

            form.addEventListener('submit', async (event) => {
                if (!validated) {
                    event.preventDefault();
                    const ok = await validateId();
                    if (ok) {
                        form.submit();
                    }
                }
            });

            updateWordCounter();
        })();
        </script>
    </body>
    </html>
    """

    return html.replace("__MIN_WORDS__", str(MIN_RESPONSE_WORDS))

@app.post("/submit", include_in_schema=False, response_class=HTMLResponse)
def handle_submission(student_id: str = Form(...), text: str = Form(...)):
    try:
        validate_student_id(student_id)
    except HTTPException as exc:
        return HTMLResponse(
            f"""
            <p>{html.escape(str(exc.detail))}</p>
            <p><a href="/submit">Go back</a></p>
            """,
            status_code=exc.status_code,
        )

    cleaned = text.strip()
    if not cleaned:
        return HTMLResponse(
            """
            <p>Please enter student's text before submitting.</p>
            <p><a href="/submit">Go back</a></p>
            """,
            status_code=400,
        )

    result = analyze_student_text(cleaned)
    print("\n=== WEB FORM SUBMISSION ===")
    display_scores(result)
    record_analysis("web-form", cleaned, result, student_id=student_id)
    alert_info = result.get("alert_metadata")
    if alert_info:
        print(
            "\n!!! HIGH-RISK ALERT TRIGGERED !!!\n"
            f"Encrypted message saved to: {alert_info.get('ciphertext_path')}\n"
            f"Key stored at: {alert_info.get('key_path')} (share only with authorized clinicians)\n"
            "Send the .enc file to the college counselor/therapist and keep the .key file secure.\n"
            "To share, email the .enc file and deliver the key via a secure channel (phone/in person)."
        )

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Submission received</title>
        <style>
            body {font-family: Arial, sans-serif; background:#f1f1f1; padding:40px;}
            .card {max-width:520px; margin:0 auto; background:white; padding:30px; border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.08); text-align:center;}
            a {display:inline-block; margin-top:20px; color:#145DA0; font-weight:bold; text-decoration:none;}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Submission received</h2>
            <p>The analysis has been written to the PowerShell terminal.</p>
            <a href="/submit">Analyze another response</a>
        </div>
    </body>
    </html>
    """

# CLI MODE
def run_cli():
    print(
        "\nTell us what you're experiencing right now (type 'quit' to exit)."
        "\nType or paste your full response after the prompt and press Enter."
    )
    while True:
        user_text = input("> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            print("Please enter something.")
            continue

        result = analyze_student_text(user_text)
        print("\n=== RESULTS ===")
        display_scores(result)
        record_analysis("cli", user_text, result)
        alert_info = result.get("alert_metadata")
        if alert_info:
            print(
                "\n!!! HIGH-RISK ALERT TRIGGERED !!!\n"
                f"Encrypted message saved to: {alert_info.get('ciphertext_path')}\n"
                f"Key stored at: {alert_info.get('key_path')} (share only with authorized clinicians)\n"
                "Send the .enc file to the college counselor/therapist and keep the .key file secure.\n"
                "To share, email the .enc file and deliver the key via a secure channel (phone/in person)."
            )

# FASTAPI SERVER MODE
def run_server(host: str, port: int):
    import uvicorn
    uvicorn.run("HackathonProject1_updated:app", host=host, port=port, log_level="info")

# ENTRY POINT
def main():
    global STRESS_MODEL_CANDIDATE, STRESS_BASE_MODEL, STRESS_TOKENIZER_FALLBACK
    global stress_model_name, stress_tokenizer, stress_model

    parser = argparse.ArgumentParser(description="Student wellness analyzer")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server instead of the CLI.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    # Fine-tuning options
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune the stress model before running inference.",
    )
    parser.add_argument(
        "--train-file",
        help="Path to the JSON/CSV training file.",
    )
    parser.add_argument(
        "--eval-file",
        help="Optional JSON/CSV validation file for evaluation during training.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column in the dataset that contains the student text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column that holds the target label.",
    )
    parser.add_argument(
        "--label-list",
        help="Comma-separated label names to enforce ordering (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default=STRESS_MODEL_CANDIDATE,
        help="Directory where the fine-tuned stress model will be stored.",
    )
    parser.add_argument(
        "--base-model",
        default=STRESS_BASE_MODEL,
        help="Base checkpoint to fine-tune (e.g., roberta-large).",
    )
    parser.add_argument(
        "--epochs", type=float, default=3.0, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Per-device training batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="AdamW learning rate."
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Tokenizer max sequence length."
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run fine-tuning and exit without launching CLI/server.",
    )
    parser.add_argument(
        "--train-csv",
        action="store_true",
        help="Load the training/eval files with pandas (CSV inputs).",
    )
    parser.add_argument(
        "--dataset-root",
        help="Path to the StudentLife dataset root (auto-generate text/stress pairs).",
    )
    parser.add_argument(
        "--pair-window",
        type=int,
        default=600,
        help="Maximum seconds between a comment and stress entry when pairing StudentLife data.",
    )
    parser.add_argument(
        "--augment-file",
        action="append",
        help="Additional CSV/JSON files with text/label columns to append to the training set. Can be passed multiple times.",
    )
    args = parser.parse_args()

    generated_csv_paths: List[Path] = []

    def _write_temp_training_file(dataframe: pd.DataFrame) -> str:
        fd, tmp_path = tempfile.mkstemp(prefix="studentlife_data_", suffix=".csv")
        os.close(fd)
        path = Path(tmp_path)
        dataframe.to_csv(path, index=False)
        generated_csv_paths.append(path)
        return str(path)

    prepared_dataframe: Optional[pd.DataFrame] = None

    try:
        if args.train and args.dataset_root:
            studentlife_df = _build_studentlife_dataframe(
                args.dataset_root, window_seconds=args.pair_window
            )
            if studentlife_df.empty:
                parser.error(
                    f"No StudentLife text samples were extracted from {args.dataset_root}."
                )

            rename_map = {}
            if args.text_column != "text":
                rename_map["text"] = args.text_column
            if args.label_column != "label":
                rename_map["label"] = args.label_column
            if rename_map:
                studentlife_df = studentlife_df.rename(columns=rename_map)

            prepared_dataframe = studentlife_df
            logger.info(
                "Built %d StudentLife samples from %s (window=%ds).",
                len(studentlife_df),
                args.dataset_root,
                args.pair_window,
            )
        elif args.train and args.dataset_root is None and args.augment_file:
            if not args.train_file:
                parser.error("--augment-file requires --train-file PATH")

        if args.train and args.augment_file:
            if prepared_dataframe is None:
                if not args.train_file:
                    parser.error("--augment-file requires --train-file PATH")
                base_df = _load_text_label_dataframe(
                    args.train_file, args.text_column, args.label_column
                )
            else:
                base_df = prepared_dataframe
            prepared_dataframe = _append_augmented_samples(
                base_df, args.augment_file, args.text_column, args.label_column
            )

        if args.train and prepared_dataframe is not None:
            args.train_file = _write_temp_training_file(prepared_dataframe)
            args.train_csv = True

        if args.train:
            if not args.train_file:
                parser.error("--train requires --train-file PATH")
            labels = _parse_label_list(args.label_list)
            train_stress_model(
                train_file=args.train_file,
                eval_file=args.eval_file,
                text_column=args.text_column,
                label_column=args.label_column,
                output_dir=args.output_dir,
                base_model_name=args.base_model,
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                label_list=labels,
                from_csv=args.train_csv,
            )
            STRESS_MODEL_CANDIDATE = args.output_dir
            STRESS_BASE_MODEL = args.base_model
            STRESS_TOKENIZER_FALLBACK = args.base_model
            (
                stress_model_name,
                stress_tokenizer,
                stress_model,
            ) = _load_stress_bundle(STRESS_MODEL_CANDIDATE, STRESS_BASE_MODEL)
            if args.train_only:
                raise SystemExit("Training completed. Saved model to " + args.output_dir)
        if args.serve:
            run_server(args.host, args.port)
        else:
            run_cli()
    finally:
        for csv_path in generated_csv_paths:
            if csv_path.exists():
                csv_path.unlink()

if __name__ == "__main__":
    main()
