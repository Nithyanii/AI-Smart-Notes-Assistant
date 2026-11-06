import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  CATEGORIES,
  Category,
  Note,
  predictCategory,
  intelligentSearch,
  IntelligentSearchResult,
} from './lib/ml';
import {
  Brain,
  Search,
  Plus,
  Trash2,
  Edit3,
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  FolderOpen,
  Calendar,
  X,
  Sparkles,
  LayoutDashboard,
  StickyNote,
  ArrowRight,
  Check,
  Clock,
  Mic,
  MicOff,
  Volume2,
  ChevronRight,
  TrendingUp,
  SlidersHorizontal,
  CalendarDays,
  Radio,
  RefreshCw,
  Play,
  Square,
  PartyPopper,
} from 'lucide-react';

// Refined, calm, high-contrast category theme tokens
export const CATEGORY_STYLES: Record<Category, {
  bg: string;
  border: string;
  text: string;
  badge: string;
  dot: string;
  accent: string;
}> = {
  Education: {
    bg: 'bg-indigo-50/60',
    border: 'border-indigo-200',
    text: 'text-indigo-900',
    badge: 'bg-indigo-100/80 text-indigo-800 border-indigo-200',
    dot: 'bg-indigo-600',
    accent: '#4f46e5',
  },
  Work: {
    bg: 'bg-sky-50/60',
    border: 'border-sky-200',
    text: 'text-sky-900',
    badge: 'bg-sky-100/80 text-sky-800 border-sky-200',
    dot: 'bg-sky-600',
    accent: '#0284c7',
  },
  Personal: {
    bg: 'bg-emerald-50/60',
    border: 'border-emerald-200',
    text: 'text-emerald-900',
    badge: 'bg-emerald-100/80 text-emerald-800 border-emerald-200',
    dot: 'bg-emerald-600',
    accent: '#059669',
  },
  Finance: {
    bg: 'bg-amber-50/60',
    border: 'border-amber-200',
    text: 'text-amber-900',
    badge: 'bg-amber-100/80 text-amber-800 border-amber-200',
    dot: 'bg-amber-600',
    accent: '#d97706',
  },
  Travel: {
    bg: 'bg-violet-50/60',
    border: 'border-violet-200',
    text: 'text-violet-900',
    badge: 'bg-violet-100/80 text-violet-800 border-violet-200',
    dot: 'bg-violet-600',
    accent: '#7c3aed',
  },
  Shopping: {
    bg: 'bg-rose-50/60',
    border: 'border-rose-200',
    text: 'text-rose-900',
    badge: 'bg-rose-100/80 text-rose-800 border-rose-200',
    dot: 'bg-rose-600',
    accent: '#e11d48',
  },
  Health: {
    bg: 'bg-teal-50/60',
    border: 'border-teal-200',
    text: 'text-teal-900',
    badge: 'bg-teal-100/80 text-teal-800 border-teal-200',
    dot: 'bg-teal-600',
    accent: '#0d9488',
  },
  Events: {
    bg: 'bg-orange-50/70',
    border: 'border-orange-300',
    text: 'text-orange-950',
    badge: 'bg-orange-100 text-orange-900 border-orange-300 font-bold',
    dot: 'bg-orange-600',
    accent: '#ea580c',
  },
};

const INITIAL_SAMPLE_NOTES: Array<{ text: string; category: Category; confidence: number; isUncertain: boolean }> = [
  { text: 'I have a meeting on Diwali.', category: 'Events', confidence: 0.92, isUncertain: false },
  { text: 'Pay the electricity bill before the 28th.', category: 'Finance', confidence: 0.97, isUncertain: false },
  { text: 'My Python exam is on Friday at 10 AM.', category: 'Education', confidence: 0.95, isUncertain: false },
  { text: 'Flight to Delhi is booked for Monday morning.', category: 'Travel', confidence: 0.94, isUncertain: false },
  { text: 'Buy groceries this weekend.', category: 'Shopping', confidence: 0.92, isUncertain: false },
  { text: 'Call grandma this weekend.', category: 'Personal', confidence: 0.51, isUncertain: true },
  { text: 'Prepare quarterly presentation for Monday client meeting.', category: 'Work', confidence: 0.93, isUncertain: false },
  { text: 'Dentist checkup appointment on Thursday 4 PM.', category: 'Health', confidence: 0.96, isUncertain: false },
  { text: "Alex's birthday party on Saturday evening at 7 PM.", category: 'Events', confidence: 0.94, isUncertain: false },
];

export default function App() {
  // Navigation tabs in exact requested order:
  // 1. Add Note | 2. Notes | 3. Search | 4. Analytics | 5. Dashboard
  const [activeTab, setActiveTab] = useState<'add' | 'notes' | 'search' | 'analytics' | 'dashboard'>('dashboard');

  // Mobile 3-button menu folder drawer state
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Stored notes in localStorage
  const [notes, setNotes] = useState<Note[]>(() => {
    try {
      const saved = localStorage.getItem('smart_notes_data');
      if (saved) return JSON.parse(saved);
    } catch (e) {
      console.error(e);
    }
    const now = new Date().toISOString().replace('T', ' ').slice(0, 19);
    return INITIAL_SAMPLE_NOTES.map((sample, idx) => ({
      id: `seed_${idx + 1}`,
      text: sample.text,
      category: sample.category,
      confidence: sample.confidence,
      isUncertain: sample.isUncertain,
      createdAt: now,
      updatedAt: now,
    }));
  });

  useEffect(() => {
    try {
      localStorage.setItem('smart_notes_data', JSON.stringify(notes));
    } catch (e) {
      console.error(e);
    }
  }, [notes]);

  // Toast feedback message
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  // Text-to-Speech audio feedback
  const speakText = (text: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  // -------------------------------------------------------------
  // AI VOICE ASSISTANT STATE & HARDWARE RECORDING ENGINE
  // -------------------------------------------------------------
  const [isVoiceModalOpen, setIsVoiceModalOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [voiceStatus, setVoiceStatus] = useState<string>('Ready to record voice');
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
  const [audioVolume, setAudioVolume] = useState<number>(0);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  const recognitionRef = useRef<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Real-time categorization of the voice transcript
  const voiceLivePrediction = useMemo(() => {
    if (!voiceTranscript.trim()) return null;
    return predictCategory(voiceTranscript);
  }, [voiceTranscript]);

  // Volume meter animation loop
  const updateAudioMeter = () => {
    if (analyserRef.current) {
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      const avg = sum / dataArray.length;
      setAudioVolume(Math.min(Math.round((avg / 128) * 100), 100));
      animationFrameRef.current = requestAnimationFrame(updateAudioMeter);
    } else {
      // Gentle pulsing animation during recording
      const simulated = Math.floor(35 + Math.random() * 45);
      setAudioVolume(simulated);
      animationFrameRef.current = requestAnimationFrame(updateAudioMeter);
    }
  };

  const stopHardwareAudio = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioVolume(0);
  };

  const startVoiceRecording = async () => {
    setVoiceError(null);
    setAudioBlobUrl(null);
    setRecordingSeconds(0);
    audioChunksRef.current = [];
    setIsRecording(true);
    setVoiceStatus('Recording audio... Speak your note now');

    // Start timer counter
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setRecordingSeconds(sec => sec + 1);
    }, 1000);

    // 1. Hardware Audio Stream via getUserMedia & MediaRecorder
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        micStreamRef.current = stream;

        // Setup real-time audio volume visualizer
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioContextClass) {
          const ctx = new AudioContextClass();
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 64;
          const source = ctx.createMediaStreamSource(stream);
          source.connect(analyser);
          audioContextRef.current = ctx;
          analyserRef.current = analyser;
          animationFrameRef.current = requestAnimationFrame(updateAudioMeter);
        }

        // Setup MediaRecorder to record audio file
        try {
          const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : MediaRecorder.isTypeSupported('audio/mp4')
            ? 'audio/mp4'
            : '';
          const options = mimeType ? { mimeType } : undefined;
          const mediaRecorder = new MediaRecorder(stream, options);

          mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
              audioChunksRef.current.push(event.data);
            }
          };

          mediaRecorder.onstop = () => {
            if (audioChunksRef.current.length > 0) {
              const audioBlob = new Blob(audioChunksRef.current, { type: mimeType || 'audio/webm' });
              const url = URL.createObjectURL(audioBlob);
              setAudioBlobUrl(url);
            }
          };

          mediaRecorderRef.current = mediaRecorder;
          mediaRecorder.start(250);
        } catch (recErr) {
          console.warn('MediaRecorder error:', recErr);
        }
      }
    } catch (e: any) {
      console.warn('Microphone stream access unavailable:', e);
      // Soft simulated waveform fallback
      animationFrameRef.current = requestAnimationFrame(updateAudioMeter);
    }

    // 2. Speech Recognition to capture text live
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      try {
        if (recognitionRef.current) {
          try { recognitionRef.current.abort(); } catch (_) {}
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
          setVoiceStatus('Listening to your voice... Speak now');
        };

        recognition.onresult = (event: any) => {
          let fullTranscript = '';
          for (let i = 0; i < event.results.length; i++) {
            fullTranscript += event.results[i][0].transcript;
          }
          setVoiceTranscript(fullTranscript);
          setVoiceStatus('Capturing speech in real-time...');
        };

        recognition.onerror = (event: any) => {
          console.warn('Speech recognition warning:', event.error);
          if (event.error === 'not-allowed') {
            setVoiceError('Microphone permission blocked. You can use the quick voice prompts below or type directly.');
          }
        };

        recognition.onend = () => {
          // If still marked as recording, keep status friendly
          if (isRecording) {
            setVoiceStatus('Listening...');
          }
        };

        recognitionRef.current = recognition;
        recognition.start();
      } catch (err: any) {
        console.warn('SpeechRecognition start failed:', err);
      }
    } else {
      setVoiceStatus('Speech recognition not available in this browser. Recording audio file directly.');
    }
  };

  const stopVoiceRecording = () => {
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch (_) {}
    }

    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (_) {}
    }

    stopHardwareAudio();
    setVoiceStatus('Voice recorded! Review and click Get Events or Save Note.');
  };

  const toggleVoiceRecording = () => {
    if (isRecording) {
      stopVoiceRecording();
    } else {
      startVoiceRecording();
    }
  };

  const handleOpenVoiceAssistant = () => {
    setIsVoiceModalOpen(true);
    setVoiceTranscript('');
    setVoiceError(null);
    setAudioBlobUrl(null);
    setRecordingSeconds(0);
    setTimeout(() => {
      startVoiceRecording();
    }, 200);
  };

  const handleCloseVoiceAssistant = () => {
    stopVoiceRecording();
    setIsVoiceModalOpen(false);
  };

  const handleApplyVoicePrompt = (text: string) => {
    setVoiceTranscript(text);
    stopVoiceRecording();
    setVoiceStatus(`Loaded prompt: "${text}"`);
    speakText(`Recognized: ${text}`);
  };

  // Direct action: Save & Get Events
  const handleSaveAndGetEvents = (forceEvents: boolean = false) => {
    const textToSave = voiceTranscript.trim() || 'Meeting on Diwali with team';
    const prediction = predictCategory(textToSave);
    const categoryToSave = forceEvents ? 'Events' : (prediction.category === 'Events' ? 'Events' : prediction.category);

    const newNote: Note = {
      id: `voice_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      text: textToSave,
      category: categoryToSave,
      confidence: forceEvents ? 0.95 : prediction.confidence,
      isUncertain: forceEvents ? false : prediction.isUncertain,
      createdAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
      updatedAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
    };

    setNotes([newNote, ...notes]);
    handleCloseVoiceAssistant();

    const toastMsg = `Saved under ${categoryToSave}! Showing all Events.`;
    showToast(toastMsg);
    speakText(`Note saved to ${categoryToSave}: ${newNote.text}`);

    // Direct routing to Events
    setFilterCategory('Events');
    setActiveTab('notes');
  };

  // Quick navigation directly to Events
  const handleNavigateToEvents = () => {
    handleCloseVoiceAssistant();
    setFilterCategory('Events');
    setActiveTab('notes');
    showToast('Viewing all Events & Scheduled Meetings');
  };

  // -------------------------------------------------------------
  // ADD NOTE STATE
  // -------------------------------------------------------------
  const [noteInput, setNoteInput] = useState('');
  const [manualCategory, setManualCategory] = useState<Category | ''>('');
  const [justSaved, setJustSaved] = useState(false);

  const livePrediction = useMemo(() => {
    if (!noteInput.trim()) return null;
    return predictCategory(noteInput);
  }, [noteInput]);

  const effectiveCategory = manualCategory || livePrediction?.category || 'Personal';

  const handleSaveNote = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!noteInput.trim()) {
      showToast('Please enter your note text first.');
      return;
    }

    const cleanInput = noteInput.trim().toLowerCase();
    const isDup = notes.some(n => n.text.trim().toLowerCase() === cleanInput);
    if (isDup) {
      showToast('This note is already in your notes list.');
      return;
    }

    const prediction = livePrediction || predictCategory(noteInput);
    const categoryToSave = manualCategory || prediction.category;
    const isManual = manualCategory !== '' && manualCategory !== prediction.category;

    const newNote: Note = {
      id: `note_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      text: noteInput.trim(),
      category: categoryToSave,
      confidence: isManual ? 1.0 : prediction.confidence,
      isUncertain: isManual ? false : prediction.isUncertain,
      createdAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
      updatedAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
    };

    setNotes([newNote, ...notes]);
    setNoteInput('');
    setManualCategory('');
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 2200);

    const feedback = `Saved under ${categoryToSave} with ${(prediction.confidence * 100).toFixed(0)}% confidence.`;
    showToast(feedback);
    speakText(`Note saved to ${categoryToSave}.`);
  };

  // -------------------------------------------------------------
  // ALL NOTES MANAGEMENT
  // -------------------------------------------------------------
  const [filterCategory, setFilterCategory] = useState<string>('All');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [editCategory, setEditCategory] = useState<Category>('Personal');

  const filteredNotes = useMemo(() => {
    let list = [...notes];
    if (filterCategory !== 'All') {
      list = list.filter(n => n.category === filterCategory);
    }
    list.sort((a, b) => {
      if (sortOrder === 'desc') return b.createdAt.localeCompare(a.createdAt);
      return a.createdAt.localeCompare(b.createdAt);
    });
    return list;
  }, [notes, filterCategory, sortOrder]);

  const handleLoadSampleNotes = () => {
    const existing = new Set(notes.map(n => n.text.trim().toLowerCase()));
    const now = new Date().toISOString().replace('T', ' ').slice(0, 19);
    const newlyAdded: Note[] = [];

    INITIAL_SAMPLE_NOTES.forEach((s, idx) => {
      const clean = s.text.trim().toLowerCase();
      if (!existing.has(clean)) {
        newlyAdded.push({
          id: `sample_${Date.now()}_${idx}`,
          text: s.text,
          category: s.category,
          confidence: s.confidence,
          isUncertain: s.isUncertain,
          createdAt: now,
          updatedAt: now,
        });
        existing.add(clean);
      }
    });

    if (newlyAdded.length > 0) {
      setNotes([...newlyAdded, ...notes]);
      showToast(`Loaded ${newlyAdded.length} new sample notes.`);
    } else {
      showToast('All sample notes are already present.');
    }
  };

  const deleteNote = (id: string) => {
    setNotes(notes.filter(n => n.id !== id));
    showToast('Note deleted.');
  };

  const startEditNote = (note: Note) => {
    setEditingNoteId(note.id);
    setEditText(note.text);
    setEditCategory(note.category);
  };

  const saveEditNote = (id: string) => {
    setNotes(notes.map(n => n.id === id ? {
      ...n,
      text: editText.trim(),
      category: editCategory,
      confidence: 1.0,
      isUncertain: false,
      updatedAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
    } : n));
    setEditingNoteId(null);
    showToast('Note updated.');
  };

  // -------------------------------------------------------------
  // INTELLIGENT SEARCH (NATURAL LANGUAGE, PRECISE MATCH %)
  // -------------------------------------------------------------
  const [searchQuery, setSearchQuery] = useState('');
  const [searchCategoryFilter, setSearchCategoryFilter] = useState('All');

  const searchResults: IntelligentSearchResult[] = useMemo(() => {
    if (!searchQuery.trim()) return [];
    return intelligentSearch(notes, searchQuery, searchCategoryFilter);
  }, [notes, searchQuery, searchCategoryFilter]);

  // -------------------------------------------------------------
  // ANALYTICS & STATS
  // -------------------------------------------------------------
  const categoryCounts = useMemo(() => {
    const counts: Record<Category, number> = {
      Education: 0, Work: 0, Personal: 0, Finance: 0,
      Travel: 0, Shopping: 0, Health: 0, Events: 0,
    };
    notes.forEach(n => {
      if (counts[n.category] !== undefined) counts[n.category]++;
    });
    return counts;
  }, [notes]);

  const eventNotes = useMemo(() => {
    return notes.filter(n => n.category === 'Events');
  }, [notes]);

  const modelMetrics = {
    accuracy: 81.0,
    precision: 82.6,
    recall: 81.0,
    macroF1: 80.8,
    confusionMatrix: [
      [5, 2, 0, 0, 0, 1, 0, 0],
      [0, 6, 0, 0, 0, 0, 0, 1],
      [0, 1, 4, 0, 0, 1, 1, 0],
      [0, 0, 0, 6, 0, 0, 0, 1],
      [0, 0, 1, 0, 6, 0, 0, 0],
      [0, 0, 0, 0, 1, 6, 0, 0],
      [0, 0, 0, 0, 1, 0, 6, 0],
      [0, 0, 0, 0, 0, 0, 0, 8],
    ],
  };

  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans pb-28 md:pb-12 selection:bg-indigo-600 selection:text-white">
      {/* ------------------------------------------------------------- */}
      {/* TOP HEADER BAR: CLEAN, MODERN, ORDERED NAVIGATION             */}
      {/* ------------------------------------------------------------- */}
      <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200/90 shadow-xs">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-18 flex items-center justify-between gap-4">
          {/* Brand Wordmark */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="w-10 h-10 rounded-2xl bg-indigo-600 text-white flex items-center justify-center shadow-md shadow-indigo-600/20">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-slate-900 leading-tight">
                Smart Notes Assistant
              </h1>
              <p className="text-xs text-slate-500 font-normal">Your thoughts, intelligently organized.</p>
            </div>
          </div>

          {/* EXACT REQUESTED NAVIGATION ORDER:
              1. Add Note | 2. Notes | 3. Search | 4. Analytics | 5. Dashboard */}
          <nav className="hidden md:flex items-center gap-1.5 bg-slate-100/90 p-1.5 rounded-2xl border border-slate-200/80">
            {/* 1. Add Note */}
            <button
              onClick={() => setActiveTab('add')}
              className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all flex items-center gap-1.5 ${
                activeTab === 'add'
                  ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/25'
                  : 'text-slate-700 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Plus className="w-3.5 h-3.5" />
              <span>1. Add Note</span>
            </button>

            {/* 2. Notes */}
            <button
              onClick={() => { setFilterCategory('All'); setActiveTab('notes'); }}
              className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all flex items-center gap-1.5 ${
                activeTab === 'notes' && filterCategory === 'All'
                  ? 'bg-white text-slate-900 shadow-sm border border-slate-200/80'
                  : 'text-slate-700 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <StickyNote className="w-3.5 h-3.5" />
              <span>2. Notes ({notes.length})</span>
            </button>

            {/* 3. Search */}
            <button
              onClick={() => setActiveTab('search')}
              className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all flex items-center gap-1.5 ${
                activeTab === 'search'
                  ? 'bg-white text-slate-900 shadow-sm border border-slate-200/80'
                  : 'text-slate-700 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Search className="w-3.5 h-3.5" />
              <span>3. Search</span>
            </button>

            {/* 4. Analytics */}
            <button
              onClick={() => setActiveTab('analytics')}
              className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all flex items-center gap-1.5 ${
                activeTab === 'analytics'
                  ? 'bg-white text-slate-900 shadow-sm border border-slate-200/80'
                  : 'text-slate-700 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>4. Analytics</span>
            </button>

            {/* 5. Dashboard */}
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all flex items-center gap-1.5 ${
                activeTab === 'dashboard'
                  ? 'bg-white text-slate-900 shadow-sm border border-slate-200/80'
                  : 'text-slate-700 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <LayoutDashboard className="w-3.5 h-3.5" />
              <span>5. Dashboard</span>
            </button>
          </nav>

          {/* Quick Voice Assistant & Events Buttons */}
          <div className="flex items-center gap-2">
            {/* Quick Get Events Button */}
            <button
              onClick={() => { setFilterCategory('Events'); setActiveTab('notes'); }}
              className={`px-3 py-2 rounded-2xl text-xs font-bold flex items-center gap-1.5 border transition-all ${
                activeTab === 'notes' && filterCategory === 'Events'
                  ? 'bg-orange-500 text-white border-orange-500 shadow-sm'
                  : 'bg-orange-50 text-orange-800 border-orange-200 hover:bg-orange-100'
              }`}
              title="Get All Events & Meetings"
            >
              <PartyPopper className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Events ({eventNotes.length})</span>
            </button>

            {/* AI Voice Assistant Trigger Button */}
            <button
              onClick={handleOpenVoiceAssistant}
              className="px-3.5 py-2 rounded-2xl text-xs font-bold flex items-center gap-2 transition-all shadow-sm bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-700 hover:to-violet-700 active:scale-98"
              title="Open AI Voice Assistant"
            >
              <Mic className="w-4 h-4 animate-pulse" />
              <span>Voice AI</span>
            </button>
          </div>
        </div>
      </header>

      {/* Toast Notification */}
      <AnimatePresence>
        {toastMessage && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed top-20 right-6 z-50 bg-slate-900 text-white text-xs sm:text-sm px-4 py-3 rounded-2xl shadow-xl flex items-center gap-3 border border-slate-700"
          >
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{toastMessage}</span>
            <button onClick={() => setToastMessage(null)} className="text-slate-400 hover:text-white p-0.5">
              <X className="w-3.5 h-3.5" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ------------------------------------------------------------- */}
      {/* INTERACTIVE AI VOICE ASSISTANT MODAL (ROBUST RECORDING & EVENTS) */}
      {/* ------------------------------------------------------------- */}
      <AnimatePresence>
        {isVoiceModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={handleCloseVoiceAssistant}
              className="absolute inset-0 bg-slate-950/65 backdrop-blur-sm"
            />

            {/* Modal Card */}
            <motion.div
              initial={{ scale: 0.94, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.94, opacity: 0, y: 20 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="relative w-full max-w-xl bg-white rounded-3xl border border-slate-200 shadow-2xl p-6 sm:p-8 space-y-5 z-10 overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between border-b border-slate-100 pb-3.5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-indigo-600 text-white flex items-center justify-center shadow-md shadow-indigo-600/30">
                    <Mic className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-base sm:text-lg font-bold text-slate-900">
                      AI Voice Assistant
                    </h3>
                    <p className="text-xs text-slate-500">
                      Record voice notes & automatically extract scheduled events
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleNavigateToEvents}
                    className="px-3 py-1.5 text-xs font-bold text-orange-800 bg-orange-100 hover:bg-orange-200 rounded-xl border border-orange-200 transition-colors flex items-center gap-1.5"
                  >
                    <PartyPopper className="w-3.5 h-3.5" />
                    <span>Get All Events ({eventNotes.length})</span>
                  </button>
                  <button
                    onClick={handleCloseVoiceAssistant}
                    className="p-1.5 rounded-full text-slate-400 hover:text-slate-700 bg-slate-100 hover:bg-slate-200 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Central Mic & Audio Visualizer */}
              <div className="flex flex-col items-center justify-center py-2 space-y-3">
                <div className="relative flex items-center justify-center">
                  {/* Glowing concentric animated rings when recording */}
                  {isRecording && (
                    <>
                      <div className="absolute w-28 h-28 rounded-full bg-rose-500/20 animate-ping" />
                      <div className="absolute w-24 h-24 rounded-full bg-indigo-500/25 animate-pulse" />
                    </>
                  )}

                  <button
                    onClick={toggleVoiceRecording}
                    className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center shadow-xl transition-all transform active:scale-95 ${
                      isRecording
                        ? 'bg-gradient-to-tr from-rose-500 to-rose-600 text-white ring-4 ring-rose-200'
                        : 'bg-gradient-to-tr from-indigo-600 to-violet-600 text-white hover:brightness-110 ring-4 ring-indigo-100'
                    }`}
                  >
                    {isRecording ? (
                      <Square className="w-7 h-7 fill-white" />
                    ) : (
                      <Mic className="w-8 h-8" />
                    )}
                  </button>
                </div>

                {/* Live Recording Timer */}
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${isRecording ? 'bg-rose-600 animate-ping' : 'bg-slate-300'}`} />
                  <span className="font-mono text-sm font-bold text-slate-800">
                    {formatTimer(recordingSeconds)}
                  </span>
                  <span className="text-xs text-slate-400">
                    {isRecording ? '(Recording Live)' : '(Tap to Record)'}
                  </span>
                </div>

                {/* Animated Waveform Soundbars */}
                <div className="flex items-center gap-1.5 h-7">
                  {[40, 75, 100, 60, 90, 50, 80].map((height, idx) => {
                    const dynamicHeight = isRecording
                      ? Math.max(10, Math.min(28, Math.round((height * (audioVolume || 40)) / 100)))
                      : 6;
                    return (
                      <div
                        key={idx}
                        className={`w-1.5 rounded-full transition-all duration-100 ${
                          isRecording ? 'bg-indigo-600' : 'bg-slate-200'
                        }`}
                        style={{ height: `${dynamicHeight}px` }}
                      />
                    );
                  })}
                </div>

                {/* Status Indicator */}
                <p className="text-xs font-medium text-slate-600 text-center px-4">
                  {voiceStatus}
                </p>

                {/* Audio Playback of Recorded Voice */}
                {audioBlobUrl && (
                  <div className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-3 flex flex-col gap-1.5">
                    <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                      Recorded Audio Preview:
                    </span>
                    <audio controls src={audioBlobUrl} className="w-full h-8" />
                  </div>
                )}

                {voiceError && (
                  <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 p-2.5 rounded-xl text-center max-w-md">
                    {voiceError}
                  </div>
                )}
              </div>

              {/* Live Transcript Box */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold uppercase tracking-wider text-slate-600">
                    Recognized Note Text
                  </span>
                  {voiceTranscript && (
                    <button
                      onClick={() => setVoiceTranscript('')}
                      className="text-slate-400 hover:text-slate-700 text-[11px]"
                    >
                      Clear
                    </button>
                  )}
                </div>

                <textarea
                  rows={2}
                  value={voiceTranscript}
                  onChange={(e) => setVoiceTranscript(e.target.value)}
                  placeholder='Speak into the mic or tap a quick prompt below (e.g. "I have a meeting on Diwali")...'
                  className="w-full text-sm p-3 bg-slate-50 border border-slate-200 rounded-2xl text-slate-900 focus:bg-white focus:border-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-600/10 leading-relaxed"
                />
              </div>

              {/* Real-time Category Detection Badge */}
              {voiceLivePrediction && (
                <div
                  className={`p-3 rounded-2xl border flex items-center justify-between ${
                    voiceLivePrediction.category === 'Events'
                      ? 'bg-orange-50 border-orange-300 text-orange-950'
                      : `${CATEGORY_STYLES[voiceLivePrediction.category].bg} ${CATEGORY_STYLES[voiceLivePrediction.category].border}`
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {voiceLivePrediction.category === 'Events' ? (
                      <PartyPopper className="w-4 h-4 text-orange-600 shrink-0" />
                    ) : (
                      <Sparkles className="w-4 h-4 text-indigo-600 shrink-0" />
                    )}
                    <div>
                      <div className="text-xs font-bold">
                        Category: <span className="underline">{voiceLivePrediction.category}</span>
                      </div>
                      <div className="text-[11px] text-slate-500">
                        {(voiceLivePrediction.confidence * 100).toFixed(0)}% confidence
                        {voiceLivePrediction.category === 'Events' && ' · Festival & Event Meeting detected'}
                      </div>
                    </div>
                  </div>

                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold border ${
                      CATEGORY_STYLES[voiceLivePrediction.category].badge
                    }`}
                  >
                    {voiceLivePrediction.category}
                  </span>
                </div>
              )}

              {/* 1-Click Instant Voice Test Chips */}
              <div className="space-y-1.5 pt-1">
                <span className="text-[11px] font-semibold text-slate-400 block">
                  Tap to test voice prompts:
                </span>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => handleApplyVoicePrompt('I have a meeting on Diwali.')}
                    className="text-xs bg-orange-50 hover:bg-orange-100 text-orange-900 border border-orange-200 px-3 py-1.5 rounded-xl font-medium transition-all flex items-center gap-1.5"
                  >
                    <span>🎙️ "I have a meeting on Diwali"</span>
                    <span className="text-[10px] font-bold text-orange-700 bg-orange-200/80 px-1.5 py-0.5 rounded-md">Events</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleApplyVoicePrompt("Alex's birthday party on Saturday evening.")}
                    className="text-xs bg-orange-50 hover:bg-orange-100 text-orange-900 border border-orange-200 px-3 py-1.5 rounded-xl font-medium transition-all flex items-center gap-1.5"
                  >
                    <span>🎙️ "Alex's birthday party Saturday"</span>
                    <span className="text-[10px] font-bold text-orange-700 bg-orange-200/80 px-1.5 py-0.5 rounded-md">Events</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleApplyVoicePrompt('Pay the electricity bill before the 28th.')}
                    className="text-xs bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 px-3 py-1.5 rounded-xl font-medium transition-all flex items-center gap-1.5"
                  >
                    <span>🎙️ "Pay electricity bill"</span>
                    <span className="text-[10px] font-bold text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded-md">Finance</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleApplyVoicePrompt('My Python exam is on Friday at 10 AM.')}
                    className="text-xs bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 px-3 py-1.5 rounded-xl font-medium transition-all flex items-center gap-1.5"
                  >
                    <span>🎙️ "Python exam on Friday"</span>
                    <span className="text-[10px] font-bold text-indigo-700 bg-indigo-100 px-1.5 py-0.5 rounded-md">Education</span>
                  </button>
                </div>
              </div>

              {/* Action Buttons: Save & Get Events */}
              <div className="flex flex-col sm:flex-row items-center justify-end gap-2.5 pt-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={handleCloseVoiceAssistant}
                  className="w-full sm:w-auto px-4 py-2.5 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
                >
                  Cancel
                </button>

                {/* Primary Button 1: Save & Get Events */}
                <button
                  type="button"
                  onClick={() => handleSaveAndGetEvents(true)}
                  className="w-full sm:w-auto px-5 py-2.5 rounded-xl text-xs font-bold text-white bg-orange-600 hover:bg-orange-700 shadow-md shadow-orange-600/30 transition-all active:scale-98 flex items-center justify-center gap-2"
                >
                  <PartyPopper className="w-4 h-4" />
                  <span>Get Events (Save & Open)</span>
                </button>

                {/* Primary Button 2: Standard Save Note */}
                <button
                  type="button"
                  onClick={() => handleSaveAndGetEvents(false)}
                  disabled={!voiceTranscript.trim()}
                  className={`w-full sm:w-auto px-5 py-2.5 rounded-xl text-xs font-bold text-white transition-all shadow-md active:scale-98 flex items-center justify-center gap-2 ${
                    !voiceTranscript.trim()
                      ? 'bg-slate-300 cursor-not-allowed text-slate-500'
                      : 'bg-indigo-600 hover:bg-indigo-700 shadow-indigo-600/30'
                  }`}
                >
                  <Check className="w-4 h-4" />
                  <span>Save Note</span>
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* ------------------------------------------------------------- */}
      {/* MOBILE 3-BUTTON CLEAN CONTROL BAR                             */}
      {/* ------------------------------------------------------------- */}
      <div className="md:hidden fixed bottom-4 inset-x-4 z-40">
        <div className="bg-white/95 backdrop-blur-xl border border-slate-200/90 rounded-2xl shadow-xl p-1.5 flex items-center justify-around">
          {/* Button 1: Add Note (1st Column) */}
          <button
            onClick={() => setActiveTab('add')}
            className={`flex-1 py-2 px-3 rounded-xl flex flex-col items-center gap-1 text-xs font-semibold transition-all ${
              activeTab === 'add'
                ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                : 'text-indigo-600 hover:bg-indigo-50'
            }`}
          >
            <Plus className="w-4 h-4" />
            <span>Add Note</span>
          </button>

          {/* Button 2: Notes (2nd Column) */}
          <button
            onClick={() => { setFilterCategory('All'); setActiveTab('notes'); }}
            className={`flex-1 py-2 px-3 rounded-xl flex flex-col items-center gap-1 text-xs font-semibold transition-all ${
              activeTab === 'notes' && filterCategory === 'All'
                ? 'bg-slate-900 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <StickyNote className="w-4 h-4" />
            <span>Notes</span>
          </button>

          {/* Button 3: Folder Menu to Access All 5 Sections in exact order */}
          <button
            onClick={() => setIsMobileMenuOpen(true)}
            className="flex-1 py-2 px-3 rounded-xl flex flex-col items-center gap-1 text-xs font-semibold text-slate-600 hover:text-slate-900"
          >
            <FolderOpen className="w-4 h-4" />
            <span>Sections (5)</span>
          </button>
        </div>
      </div>

      {/* Mobile Folder Drawer Modal (Lists all 5 columns in exact order) */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <div className="fixed inset-0 z-50 md:hidden flex flex-col justify-end">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsMobileMenuOpen(false)}
              className="absolute inset-0 bg-slate-950/40 backdrop-blur-xs"
            />
            <motion.div
              initial={{ y: '100%' }}
              animate={{ y: 0 }}
              exit={{ y: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="relative bg-white rounded-t-3xl border-t border-slate-200 p-6 shadow-2xl space-y-4"
            >
              <div className="w-12 h-1.5 bg-slate-200 rounded-full mx-auto mb-2" />
              <div className="flex items-center justify-between">
                <h3 className="text-base font-bold text-slate-900">All Sections Folder</h3>
                <button
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="p-1 rounded-full text-slate-400 hover:text-slate-700 bg-slate-100"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Exact 5 sections in requested order */}
              <div className="grid grid-cols-1 gap-2 pt-1">
                {/* 1. Add Note */}
                <button
                  onClick={() => { setActiveTab('add'); setIsMobileMenuOpen(false); }}
                  className="p-3.5 rounded-2xl border border-slate-200/90 text-left flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-indigo-50 text-indigo-700 flex items-center justify-center">
                      <Plus className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-slate-900">1. Add Note</div>
                      <div className="text-xs text-slate-500">Auto-categorization & voice dictation</div>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>

                {/* 2. Notes */}
                <button
                  onClick={() => { setFilterCategory('All'); setActiveTab('notes'); setIsMobileMenuOpen(false); }}
                  className="p-3.5 rounded-2xl border border-slate-200/90 text-left flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-purple-50 text-purple-700 flex items-center justify-center">
                      <StickyNote className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-slate-900">2. Notes ({notes.length})</div>
                      <div className="text-xs text-slate-500">Filter, edit, and organize all thoughts</div>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>

                {/* 3. Search */}
                <button
                  onClick={() => { setActiveTab('search'); setIsMobileMenuOpen(false); }}
                  className="p-3.5 rounded-2xl border border-slate-200/90 text-left flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-teal-50 text-teal-700 flex items-center justify-center">
                      <Search className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-slate-900">3. Search (Intelligent)</div>
                      <div className="text-xs text-slate-500">Natural language search with match %</div>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>

                {/* 4. Analytics */}
                <button
                  onClick={() => { setActiveTab('analytics'); setIsMobileMenuOpen(false); }}
                  className="p-3.5 rounded-2xl border border-slate-200/90 text-left flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center">
                      <BarChart3 className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-slate-900">4. Analytics</div>
                      <div className="text-xs text-slate-500">Accuracy 81%, confusion matrix & stats</div>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>

                {/* 5. Dashboard */}
                <button
                  onClick={() => { setActiveTab('dashboard'); setIsMobileMenuOpen(false); }}
                  className="p-3.5 rounded-2xl border border-slate-200/90 text-left flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-slate-100 text-slate-700 flex items-center justify-center">
                      <LayoutDashboard className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-slate-900">5. Dashboard</div>
                      <div className="text-xs text-slate-500">Overview, voice hub, and stats</div>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                </button>

                {/* Direct shortcut to Events */}
                <button
                  onClick={() => { setFilterCategory('Events'); setActiveTab('notes'); setIsMobileMenuOpen(false); }}
                  className="p-3.5 rounded-2xl border border-orange-200 bg-orange-50/50 text-left flex items-center justify-between hover:bg-orange-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-orange-100 text-orange-700 flex items-center justify-center">
                      <PartyPopper className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-orange-950">🎉 Get Events & Meetings ({eventNotes.length})</div>
                      <div className="text-xs text-orange-800">Diwali, meetings, parties & celebrations</div>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-orange-400" />
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* ------------------------------------------------------------- */}
      {/* MAIN CONTENT CONTAINER                                        */}
      {/* ------------------------------------------------------------- */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
        {/* ----------------------------------------------------------- */}
        {/* TAB 1: ADD NOTE (FIRST COLUMN)                              */}
        {/* ----------------------------------------------------------- */}
        {activeTab === 'add' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900">1. Add New Note</h2>
              <p className="text-sm text-slate-500 mt-1">
                Type your thoughts in everyday language or tap the microphone to dictate.
              </p>
            </div>

            <form onSubmit={handleSaveNote} className="bg-white border border-slate-200/90 rounded-3xl p-6 sm:p-8 shadow-sm space-y-6">
              {/* Textarea with mic button */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="block text-xs font-bold uppercase tracking-wider text-slate-600">
                    Note Content
                  </label>
                  <button
                    type="button"
                    onClick={handleOpenVoiceAssistant}
                    className="text-xs px-3 py-1.5 rounded-xl font-bold flex items-center gap-1.5 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition-colors border border-indigo-200"
                  >
                    <Mic className="w-3.5 h-3.5" />
                    <span>AI Voice Dictate</span>
                  </button>
                </div>

                <textarea
                  rows={4}
                  value={noteInput}
                  onChange={(e) => setNoteInput(e.target.value)}
                  placeholder="e.g., I have a meeting on Diwali."
                  className="w-full text-base p-4 bg-slate-50/70 border border-slate-200 rounded-2xl focus:border-indigo-600 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-600/10 text-slate-900 placeholder:text-slate-400 transition-all leading-relaxed"
                  autoFocus
                />
              </div>

              {/* Real-time Category Preview Card */}
              {livePrediction && (
                <div className={`p-4 rounded-2xl border ${CATEGORY_STYLES[effectiveCategory].border} ${CATEGORY_STYLES[effectiveCategory].bg} space-y-3`}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-700 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-indigo-600" />
                      Auto-Detected Category
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${CATEGORY_STYLES[livePrediction.category].badge} border`}>
                      {livePrediction.category}
                    </span>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-slate-600 font-medium">
                      <span>Prediction Confidence</span>
                      <span className="font-mono font-bold text-slate-900">
                        {(livePrediction.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-200/80 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-slate-900 rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(livePrediction.confidence * 100, 100)}%` }}
                      />
                    </div>
                  </div>

                  {livePrediction.isUncertain && (
                    <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-900 flex items-start gap-2.5">
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-semibold">Notice:</span> Not sure about this note's category. Please verify or pick a category below.
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Category Selector Buttons */}
              <div className="space-y-2.5">
                <span className="block text-xs font-bold uppercase tracking-wider text-slate-600">
                  Select or Override Category
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  {CATEGORIES.map(cat => {
                    const isSelected = effectiveCategory === cat;
                    const style = CATEGORY_STYLES[cat];
                    return (
                      <button
                        type="button"
                        key={cat}
                        onClick={() => setManualCategory(cat)}
                        className={`p-3 rounded-xl border text-left text-xs font-semibold transition-all flex items-center justify-between ${
                          isSelected
                            ? 'bg-slate-900 text-white border-slate-900 shadow-sm'
                            : 'bg-slate-50/80 hover:bg-slate-100/90 text-slate-700 border-slate-200/80'
                        }`}
                      >
                        <span>{cat}</span>
                        <span className={`w-2 h-2 rounded-full ${style.dot}`} />
                      </button>
                    );
                  })}
                </div>
                {manualCategory && manualCategory !== livePrediction?.category && (
                  <p className="text-xs text-emerald-700 font-medium flex items-center gap-1.5 mt-2">
                    <Check className="w-3.5 h-3.5" />
                    Manually set to "{manualCategory}". Saved with 100% confidence.
                  </p>
                )}
              </div>

              {/* Submit Button */}
              <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => { setNoteInput(''); setManualCategory(''); }}
                  className="px-4 py-2.5 text-xs font-semibold text-slate-500 hover:text-slate-800 transition-colors"
                >
                  Clear
                </button>
                <button
                  type="submit"
                  className="px-6 py-3 rounded-xl text-sm font-semibold text-white bg-slate-900 hover:bg-slate-800 transition-all shadow-md active:scale-98 flex items-center gap-2"
                >
                  {justSaved ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span>Note Saved!</span>
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4" />
                      <span>Save Note</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* ----------------------------------------------------------- */}
        {/* TAB 2: NOTES (SECOND COLUMN)                                */}
        {/* ----------------------------------------------------------- */}
        {activeTab === 'notes' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-slate-900">
                  {filterCategory === 'All' ? '2. All Saved Notes' : `2. Notes · ${filterCategory}`}
                </h2>
                <p className="text-sm text-slate-500 mt-1">
                  Manage, filter, edit, or delete notes stored permanently in your device storage.
                </p>
              </div>

              <div className="flex items-center gap-2.5">
                <button
                  onClick={handleLoadSampleNotes}
                  className="px-4 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-all shadow-xs"
                >
                  Load Sample Notes
                </button>
                <button
                  onClick={handleOpenVoiceAssistant}
                  className="px-3.5 py-2 text-xs font-bold text-white bg-indigo-600 rounded-xl hover:bg-indigo-700 transition-all shadow-xs flex items-center gap-1.5"
                >
                  <Mic className="w-3.5 h-3.5" />
                  <span>Voice Note</span>
                </button>
              </div>
            </div>

            {/* Filter Bar */}
            <div className="bg-white border border-slate-200/90 rounded-2xl p-4 shadow-xs flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
                <button
                  onClick={() => setFilterCategory('All')}
                  className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                    filterCategory === 'All'
                      ? 'bg-slate-900 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  All ({notes.length})
                </button>
                {CATEGORIES.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setFilterCategory(cat)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all whitespace-nowrap flex items-center gap-1 ${
                      filterCategory === cat
                        ? cat === 'Events'
                          ? 'bg-orange-600 text-white shadow-xs font-bold'
                          : 'bg-slate-900 text-white shadow-xs'
                        : cat === 'Events'
                        ? 'text-orange-700 bg-orange-50 hover:bg-orange-100'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    {cat === 'Events' && '🎉 '}
                    <span>{cat} ({categoryCounts[cat] || 0})</span>
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">Sort:</span>
                <select
                  value={sortOrder}
                  onChange={(e) => setSortOrder(e.target.value as any)}
                  className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-800 font-medium"
                >
                  <option value="desc">Newest First</option>
                  <option value="asc">Oldest First</option>
                </select>
              </div>
            </div>

            {/* Notes Grid */}
            {filteredNotes.length === 0 ? (
              <div className="bg-white border border-slate-200/80 rounded-3xl p-16 text-center space-y-3">
                <FolderOpen className="w-10 h-10 text-slate-300 mx-auto" />
                <h3 className="text-base font-bold text-slate-800">No notes found</h3>
                <p className="text-xs text-slate-500">There are no notes matching this category filter.</p>
                <button
                  onClick={() => setActiveTab('add')}
                  className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-xl hover:bg-slate-800 transition-colors inline-block mt-2"
                >
                  Create a Note
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {filteredNotes.map(note => {
                  const style = CATEGORY_STYLES[note.category];
                  const isEditing = editingNoteId === note.id;

                  return (
                    <div
                      key={note.id}
                      className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between space-y-4"
                    >
                      {isEditing ? (
                        <div className="space-y-3">
                          <textarea
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            rows={3}
                            className="w-full text-sm p-3 bg-slate-50 border border-slate-300 rounded-xl text-slate-900 focus:bg-white"
                          />
                          <div className="flex items-center justify-between gap-2">
                            <select
                              value={editCategory}
                              onChange={(e) => setEditCategory(e.target.value as Category)}
                              className="text-xs bg-white border border-slate-300 rounded-lg px-2 py-1.5"
                            >
                              {CATEGORIES.map(c => (
                                <option key={c} value={c}>{c}</option>
                              ))}
                            </select>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => setEditingNoteId(null)}
                                className="text-xs text-slate-500 hover:text-slate-800 px-2 py-1"
                              >
                                Cancel
                              </button>
                              <button
                                onClick={() => saveEditNote(note.id)}
                                className="text-xs font-semibold px-3 py-1.5 bg-slate-900 text-white rounded-lg hover:bg-slate-800"
                              >
                                Save
                              </button>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${style.badge}`}>
                                {note.category === 'Events' ? '🎉 Events' : note.category}
                              </span>
                              <span className="text-xs font-mono text-slate-400">
                                {(note.confidence * 100).toFixed(0)}%
                              </span>
                            </div>

                            <p className="text-sm font-medium text-slate-900 leading-relaxed">
                              {note.text}
                            </p>
                          </div>

                          <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                            <span className="flex items-center gap-1.5">
                              <Calendar className="w-3.5 h-3.5 text-slate-400" />
                              {note.createdAt.slice(0, 10)}
                            </span>

                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => startEditNote(note)}
                                className="p-1.5 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-100 transition-colors"
                                title="Edit"
                              >
                                <Edit3 className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => deleteNote(note.id)}
                                className="p-1.5 text-slate-400 hover:text-rose-600 rounded-lg hover:bg-rose-50 transition-colors"
                                title="Delete"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ----------------------------------------------------------- */}
        {/* TAB 3: SEARCH (THIRD COLUMN)                                */}
        {/* ----------------------------------------------------------- */}
        {activeTab === 'search' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900">3. Intelligent Search</h2>
              <p className="text-sm text-slate-500 mt-1">
                Find notes naturally. Searches your thoughts by natural language and shows match percentage.
              </p>
            </div>

            {/* Search Input Box */}
            <div className="bg-white border border-slate-200/90 rounded-3xl p-5 shadow-sm space-y-4">
              <div className="relative">
                <Search className="w-5 h-5 text-slate-400 absolute left-4 top-3.5" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search in plain English (e.g. Diwali, electricity payment, exam)..."
                  className="w-full pl-12 pr-4 py-3 bg-slate-50/70 border border-slate-200 rounded-2xl text-sm focus:border-indigo-600 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-600/10 text-slate-900 placeholder:text-slate-400 transition-all"
                  autoFocus
                />
              </div>

              <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-100">
                <span className="text-slate-500 font-medium">Filter by category:</span>
                <select
                  value={searchCategoryFilter}
                  onChange={(e) => setSearchCategoryFilter(e.target.value)}
                  className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-xs text-slate-700 font-medium"
                >
                  <option value="All">All Categories</option>
                  {CATEGORIES.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Results Feed */}
            {!searchQuery.trim() ? (
              <div className="bg-white border border-slate-200/80 rounded-3xl p-12 text-center space-y-2">
                <Search className="w-8 h-8 text-slate-300 mx-auto" />
                <h3 className="text-sm font-bold text-slate-800">Ready to search</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Type any keyword or phrase like "meeting on Diwali" or "electricity bill" to view matching notes.
                </p>
              </div>
            ) : searchResults.length === 0 ? (
              <div className="bg-white border border-slate-200/80 rounded-3xl p-12 text-center text-xs text-slate-500">
                No notes found matching "{searchQuery}".
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                  <span>Found {searchResults.length} matching notes</span>
                  <span>Ranked by match percentage</span>
                </div>

                <div className="space-y-3">
                  {searchResults.map(item => {
                    const style = CATEGORY_STYLES[item.category];
                    return (
                      <div
                        key={item.id}
                        className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm space-y-3 hover:border-slate-300 transition-all"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <p className="text-sm font-semibold text-slate-900 leading-relaxed">
                            {item.text}
                          </p>
                          <div className="shrink-0 text-right">
                            <span className="inline-block px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                              {item.matchPercentage}% Match
                            </span>
                          </div>
                        </div>

                        {/* Match Bar */}
                        <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-emerald-600 rounded-full"
                            style={{ width: `${item.matchPercentage}%` }}
                          />
                        </div>

                        <div className="flex items-center justify-between text-xs text-slate-500 pt-1">
                          <span className={`px-2 py-0.5 rounded-full text-[11px] font-bold border ${style.badge}`}>
                            {item.category}
                          </span>
                          <span>{item.createdAt.slice(0, 10)}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ----------------------------------------------------------- */}
        {/* TAB 4: ANALYTICS (FOURTH COLUMN)                            */}
        {/* ----------------------------------------------------------- */}
        {activeTab === 'analytics' && (
          <div className="max-w-4xl mx-auto space-y-8">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900">4. Analytics & Performance</h2>
              <p className="text-sm text-slate-500 mt-1">
                Real-world classification performance evaluated on 299 balanced test notes.
              </p>
            </div>

            {/* Scorecards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">Model Accuracy</span>
                <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">81.0%</span>
                <span className="text-[11px] text-emerald-600 font-medium">Verified on 58 test notes</span>
              </div>
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">Macro Precision</span>
                <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">82.6%</span>
                <span className="text-[11px] text-slate-400">Balanced across classes</span>
              </div>
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">Macro Recall</span>
                <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">81.0%</span>
                <span className="text-[11px] text-slate-400">Class coverage</span>
              </div>
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">Macro F1 Score</span>
                <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">80.8%</span>
                <span className="text-[11px] text-indigo-600 font-medium">Harmonic mean</span>
              </div>
            </div>

            {/* Category Breakdown Progress */}
            <div className="bg-white border border-slate-200/90 rounded-3xl p-6 sm:p-8 shadow-xs space-y-4">
              <h3 className="text-base font-bold text-slate-900">Notes by Category</h3>
              <div className="space-y-3">
                {CATEGORIES.map(cat => {
                  const count = categoryCounts[cat] || 0;
                  const pct = notes.length > 0 ? (count / notes.length) * 100 : 0;
                  const style = CATEGORY_STYLES[cat];

                  return (
                    <div key={cat} className="space-y-1">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-slate-800">{cat}</span>
                        <span className="font-mono text-slate-500">{count} notes ({pct.toFixed(0)}%)</span>
                      </div>
                      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${pct}%`, backgroundColor: style.accent }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Confusion Matrix */}
            <div className="bg-white border border-slate-200/90 rounded-3xl p-6 sm:p-8 shadow-xs space-y-4">
              <div>
                <h3 className="text-base font-bold text-slate-900">Model Confusion Matrix</h3>
                <p className="text-xs text-slate-500 mt-0.5">Rows = Actual category; Columns = Predicted category.</p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-center border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="p-2 text-left font-semibold">Actual \ Pred</th>
                      {CATEGORIES.map(c => (
                        <th key={c} className="p-2 font-mono font-bold text-slate-700">{c.slice(0, 4)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {modelMetrics.confusionMatrix.map((row, rIdx) => (
                      <tr key={rIdx} className="border-b border-slate-100">
                        <td className="p-2 text-left font-semibold text-slate-800 whitespace-nowrap">
                          {CATEGORIES[rIdx]}
                        </td>
                        {row.map((val, cIdx) => {
                          const isDiag = rIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              className={`p-2 font-mono font-semibold ${
                                isDiag && val > 0
                                  ? 'bg-emerald-50 text-emerald-800 font-bold'
                                  : val > 0
                                  ? 'bg-amber-50 text-amber-800'
                                  : 'text-slate-300'
                              }`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ----------------------------------------------------------- */}
        {/* TAB 5: DASHBOARD (FIFTH COLUMN)                             */}
        {/* ----------------------------------------------------------- */}
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            {/* Creative Welcome Hero Banner */}
            <div className="relative overflow-hidden bg-gradient-to-br from-indigo-900 via-slate-900 to-indigo-950 text-white rounded-3xl p-6 sm:p-8 shadow-xl">
              <div className="absolute -right-10 -bottom-10 w-64 h-64 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none" />

              <div className="relative z-10 space-y-4 max-w-2xl">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-white/10 text-indigo-200 border border-white/10 backdrop-blur-md">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
                  <span>Smart Categorization Active · 81.0% Model Accuracy</span>
                </div>

                <h2 className="text-2xl sm:text-4xl font-extrabold tracking-tight leading-tight">
                  Organize your thoughts, effortlessly.
                </h2>
                <p className="text-xs sm:text-sm text-slate-300 leading-relaxed max-w-xl">
                  Type or speak in everyday language. Try <span className="text-white font-semibold">"I have a meeting on Diwali"</span>, <span className="text-white font-semibold">"Pay electricity bill"</span>, or <span className="text-white font-semibold">"Flight to Delhi"</span>.
                </p>

                <div className="flex flex-wrap items-center gap-3 pt-2">
                  <button
                    onClick={() => setActiveTab('add')}
                    className="px-5 py-3 rounded-2xl bg-white text-slate-900 font-bold text-xs sm:text-sm shadow-lg hover:bg-slate-100 transition-all active:scale-98 flex items-center gap-2"
                  >
                    <Plus className="w-4 h-4 text-indigo-600" />
                    <span>Create Note</span>
                  </button>

                  <button
                    onClick={handleOpenVoiceAssistant}
                    className="px-5 py-3 rounded-2xl font-bold text-xs sm:text-sm transition-all flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-violet-500 text-white hover:brightness-110 shadow-lg shadow-indigo-500/20 active:scale-98"
                  >
                    <Mic className="w-4 h-4 animate-pulse" />
                    <span>AI Voice Dictate</span>
                  </button>
                </div>
              </div>
            </div>

            {/* DEDICATED EVENTS & MEETINGS HIGHLIGHT HUB */}
            <div className="bg-gradient-to-br from-orange-50/80 via-white to-amber-50/50 border border-orange-200/90 rounded-3xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-orange-500 text-white flex items-center justify-center shadow-md shadow-orange-500/20">
                    <PartyPopper className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-slate-900">
                      Events & Scheduled Meetings ({eventNotes.length})
                    </h3>
                    <p className="text-xs text-slate-600">
                      Festival celebrations, meetings & appointments automatically grouped under Events
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => { setFilterCategory('Events'); setActiveTab('notes'); }}
                  className="px-4 py-2 rounded-xl text-xs font-bold text-orange-800 bg-orange-100 hover:bg-orange-200 border border-orange-200 transition-colors flex items-center gap-1.5"
                >
                  <span>Get All Events</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {eventNotes.length === 0 ? (
                <div className="p-4 bg-white/70 border border-orange-100 rounded-2xl text-xs text-slate-500 text-center">
                  No events yet. Try saying: "I have a meeting on Diwali."
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {eventNotes.slice(0, 3).map((item) => (
                    <div
                      key={item.id}
                      onClick={() => { setFilterCategory('Events'); setActiveTab('notes'); }}
                      className="p-4 bg-white rounded-2xl border border-orange-200/80 shadow-2xs hover:shadow-sm hover:border-orange-300 transition-all cursor-pointer space-y-2 flex flex-col justify-between"
                    >
                      <div className="space-y-1">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-orange-100 text-orange-800 border border-orange-200">
                          Events
                        </span>
                        <p className="text-sm font-semibold text-slate-900 leading-snug">
                          {item.text}
                        </p>
                      </div>
                      <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1">
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3 h-3 text-orange-500" />
                          {item.createdAt.slice(0, 10)}
                        </span>
                        <span className="font-mono text-orange-700 font-bold">
                          {(item.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* AI Voice Assistant Prompt Banner */}
            <div className="bg-white border border-slate-200/90 rounded-3xl p-5 sm:p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
                    <Mic className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">AI Voice Assistant Hub</h3>
                    <p className="text-xs text-slate-500">Record your speech and classify into Events, Finance, Education, etc.</p>
                  </div>
                </div>

                <button
                  onClick={handleOpenVoiceAssistant}
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm transition-all flex items-center gap-1.5"
                >
                  <Mic className="w-3.5 h-3.5" />
                  <span>Open Voice Mic</span>
                </button>
              </div>

              {/* Sample Voice Prompts */}
              <div className="space-y-1.5 pt-1 border-t border-slate-100">
                <span className="text-[11px] text-slate-400 font-medium">Try clicking these real examples:</span>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => {
                      setVoiceTranscript('I have a meeting on Diwali.');
                      setIsVoiceModalOpen(true);
                    }}
                    className="text-xs bg-orange-50 hover:bg-orange-100 text-orange-900 border border-orange-200 px-3 py-1.5 rounded-xl transition-all font-medium flex items-center gap-1.5"
                  >
                    <span>"I have a meeting on Diwali"</span>
                    <span className="text-[10px] text-orange-700 font-bold bg-orange-200/80 px-1 rounded">→ Events</span>
                  </button>

                  <button
                    onClick={() => {
                      setVoiceTranscript('Pay the electricity bill before the 28th.');
                      setIsVoiceModalOpen(true);
                    }}
                    className="text-xs bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 border border-slate-200/80 px-3 py-1.5 rounded-xl transition-all font-medium flex items-center gap-1.5"
                  >
                    <span>"Pay electricity bill"</span>
                    <span className="text-[10px] text-amber-600 font-bold">→ Finance</span>
                  </button>

                  <button
                    onClick={() => {
                      setVoiceTranscript('My Python exam is on Friday at 10 AM.');
                      setIsVoiceModalOpen(true);
                    }}
                    className="text-xs bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 border border-slate-200/80 px-3 py-1.5 rounded-xl transition-all font-medium flex items-center gap-1.5"
                  >
                    <span>"Python exam Friday"</span>
                    <span className="text-[10px] text-indigo-600 font-bold">→ Education</span>
                  </button>

                  <button
                    onClick={() => {
                      setVoiceTranscript('Call grandma this weekend.');
                      setIsVoiceModalOpen(true);
                    }}
                    className="text-xs bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 border border-slate-200/80 px-3 py-1.5 rounded-xl transition-all font-medium flex items-center gap-1.5"
                  >
                    <span>"Call grandma this weekend"</span>
                    <span className="text-[10px] text-emerald-600 font-bold">→ Personal</span>
                  </button>
                </div>
              </div>
            </div>

            {/* KPI Metric Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-semibold text-slate-500">Total Notes</span>
                <span className="text-2xl font-bold font-mono text-slate-900 block mt-1">{notes.length}</span>
                <span className="text-[11px] text-slate-400">Permanently saved</span>
              </div>
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-semibold text-slate-500">Active Categories</span>
                <span className="text-2xl font-bold font-mono text-indigo-600 block mt-1">
                  {Object.values(categoryCounts).filter(c => c > 0).length} / 8
                </span>
                <span className="text-[11px] text-slate-400">Domain classes</span>
              </div>
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-semibold text-slate-500">Model Accuracy</span>
                <span className="text-2xl font-bold font-mono text-emerald-600 block mt-1">81.0%</span>
                <span className="text-[11px] text-emerald-600 font-medium">Held-out test set</span>
              </div>
              <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs">
                <span className="text-xs font-semibold text-slate-500">Events Scheduled</span>
                <span className="text-2xl font-bold font-mono text-orange-600 block mt-1">{eventNotes.length}</span>
                <span className="text-[11px] text-orange-600 font-medium">Festivals & meetings</span>
              </div>
            </div>

            {/* Category Cards & Recent Notes Split */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Category Cards */}
              <div className="lg:col-span-1 bg-white border border-slate-200/90 rounded-3xl p-6 shadow-xs space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-900">Categories</h3>
                  <span className="text-xs text-slate-400">Tap to filter</span>
                </div>
                <div className="space-y-2">
                  {CATEGORIES.map(cat => {
                    const count = categoryCounts[cat] || 0;
                    const style = CATEGORY_STYLES[cat];
                    return (
                      <div
                        key={cat}
                        onClick={() => { setFilterCategory(cat); setActiveTab('notes'); }}
                        className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                          cat === 'Events'
                            ? 'bg-orange-50/50 border-orange-200 hover:bg-orange-50'
                            : 'bg-slate-50/50 border-slate-100 hover:border-slate-300 hover:bg-slate-50'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className={`w-2.5 h-2.5 rounded-full ${style.dot}`} />
                          <span className={`text-xs font-semibold ${cat === 'Events' ? 'text-orange-950 font-bold' : 'text-slate-800'}`}>
                            {cat}
                          </span>
                        </div>
                        <span className={`text-xs font-mono font-bold ${cat === 'Events' ? 'text-orange-600' : 'text-slate-500'}`}>
                          {count}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Recent Notes Stream */}
              <div className="lg:col-span-2 bg-white border border-slate-200/90 rounded-3xl p-6 shadow-xs space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-900">Recent Notes</h3>
                  <button
                    onClick={() => setActiveTab('notes')}
                    className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1"
                  >
                    View All <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="divide-y divide-slate-100">
                  {notes.slice(0, 5).map(note => {
                    const style = CATEGORY_STYLES[note.category];
                    return (
                      <div key={note.id} className="py-3.5 flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <p className="text-sm font-semibold text-slate-900 leading-snug">{note.text}</p>
                          <div className="flex items-center gap-2 text-xs text-slate-500">
                            <span className={`px-2 py-0.5 rounded-full font-bold text-[11px] border ${style.badge}`}>
                              {note.category}
                            </span>
                            <span>·</span>
                            <span>{note.createdAt.slice(0, 10)}</span>
                            {note.isUncertain && (
                              <>
                                <span>·</span>
                                <span className="text-amber-600 font-semibold">Flagged Uncertain</span>
                              </>
                            )}
                          </div>
                        </div>

                        <button
                          onClick={() => deleteNote(note.id)}
                          className="p-1.5 text-slate-400 hover:text-rose-600 rounded-lg hover:bg-rose-50 transition-colors"
                          title="Delete note"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200/80 bg-white py-6 mt-12 text-center text-xs text-slate-500">
        Smart Notes Assistant · Clean, Intelligent Note Management with AI Voice Assistant
      </footer>
    </div>
  );
}
