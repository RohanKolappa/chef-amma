import { useState, useCallback, useEffect, useRef } from "react";
import {
  LiveKitRoom,
  useVoiceAssistant,
  BarVisualizer,
  RoomAudioRenderer,
  useConnectionState,
  useRoomContext,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { ConnectionState, RoomEvent, TranscriptionSegment, Participant } from "livekit-client";

const TOKEN_SERVER_URL = window.location.hostname === "localhost"
  ? "http://localhost:8080"
  : "";

// ── Types ────────────────────────────────────────────────────────────

interface TokenResponse {
  token: string;
  room: string;
  identity: string;
  livekit_url: string;
}

interface TranscriptEntry {
  id: string;
  speaker: "user" | "agent";
  text: string;
  isFinal: boolean;
}

// ── Voice Agent Panel ────────────────────────────────────────────────

function AgentPanel({ onDisconnect }: { onDisconnect: () => void }) {
  const { state, audioTrack } = useVoiceAssistant();
  const room = useRoomContext();
  const connectionState = useConnectionState();
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const transcriptRef = useRef<HTMLDivElement>(null);

  // Listen to ALL transcription events from the room
  useEffect(() => {
    const handleTranscription = (
      segments: TranscriptionSegment[],
      participant?: Participant
    ) => {
      for (const segment of segments) {
        const isAgent = participant?.isAgent ?? false;
        const speaker: "user" | "agent" = isAgent ? "agent" : "user";

        setTranscript((prev) => {
          const existingIdx = prev.findIndex((e) => e.id === segment.id);
          const entry: TranscriptEntry = {
            id: segment.id,
            speaker,
            text: segment.text,
            isFinal: segment.final,
          };

          if (existingIdx >= 0) {
            // Update existing segment (interim → final)
            const updated = [...prev];
            updated[existingIdx] = entry;
            return updated;
          }
          return [...prev, entry];
        });
      }
    };

    room.on(RoomEvent.TranscriptionReceived, handleTranscription);
    return () => {
      room.off(RoomEvent.TranscriptionReceived, handleTranscription);
    };
  }, [room]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  const getStateLabel = () => {
    switch (state) {
      case "connecting":
        return "Connecting to Chef Amma...";
      case "initializing":
        return "Chef Amma is preparing her kitchen...";
      case "listening":
        return "Listening...";
      case "thinking":
        return "Chef Amma is thinking...";
      case "speaking":
        return "Chef Amma is speaking...";
      default:
        return "";
    }
  };

  return (
    <div className="agent-panel">
      <div className="agent-header">
        <h2>Chef Amma</h2>
        <span className="agent-state">{getStateLabel()}</span>
      </div>

      <div className="visualizer-container">
        <BarVisualizer
          state={state}
          barCount={7}
          trackRef={audioTrack}
          options={{ minHeight: 8 }}
        />
      </div>

      <div className="transcript-container" ref={transcriptRef}>
        {transcript.length === 0 && (
          <p className="transcript-placeholder">
            Chef Amma will greet you momentarily...
          </p>
        )}
        {transcript.map((entry) => (
          <div
            key={entry.id}
            className={`transcript-entry ${entry.speaker} ${
              entry.isFinal ? "final" : "interim"
            }`}
          >
            <span className="speaker-label">
              {entry.speaker === "agent" ? "Chef Amma" : "You"}
            </span>
            <p>{entry.text}</p>
          </div>
        ))}
      </div>

      <button
        className="end-call-btn"
        onClick={onDisconnect}
        disabled={connectionState === ConnectionState.Disconnected}
      >
        End Call
      </button>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────

export default function App() {
  const [connectionDetails, setConnectionDetails] =
    useState<TokenResponse | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCall = useCallback(async () => {
    setIsConnecting(true);
    setError(null);

    try {
      const resp = await fetch(`${TOKEN_SERVER_URL}/api/token`);
      if (!resp.ok) throw new Error(`Token server error: ${resp.status}`);
      const data: TokenResponse = await resp.json();
      setConnectionDetails(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to connect to token server"
      );
      setIsConnecting(false);
    }
  }, []);

  const endCall = useCallback(() => {
    setConnectionDetails(null);
    setIsConnecting(false);
  }, []);

  // ── Landing screen ──
  if (!connectionDetails) {
    return (
      <div className="landing">
        <div className="landing-content">
          <div className="logo-icon">🍛</div>
          <h1>Chef Amma</h1>
          <p className="tagline">Your South Indian Cooking Mentor</p>
          <p className="description">
            Ask me about dosas, sambhar, rasam, or any South Indian dish.
            I'll guide you through recipes, share cooking wisdom, and help you
            find ingredients near you.
          </p>

          <button
            className="start-call-btn"
            onClick={startCall}
            disabled={isConnecting}
          >
            {isConnecting ? "Connecting..." : "Start Cooking Session"}
          </button>

          {error && <p className="error-msg">{error}</p>}
        </div>
      </div>
    );
  }

  // ── Active call ──
  return (
    <LiveKitRoom
      token={connectionDetails.token}
      serverUrl={connectionDetails.livekit_url}
      connectOptions={{ autoSubscribe: true }}
      audio={true}
      onDisconnected={endCall}
    >
      <div className="call-screen">
        <AgentPanel onDisconnect={endCall} />
      </div>
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}