"use client";

import { useRef, useEffect, useState } from "react";

// --- Custom Hook for Focus Engine Logics ---
function useFocusEngine() {
  const [score, setScore] = useState(100.0);
  const [nudgeLevel, setNudgeLevel] = useState(0);
  const [isDistracted, setIsDistracted] = useState(false);
  const [distractionSecs, setDistractionSecs] = useState(0);
  const [focusedStreak, setFocusedStreak] = useState(0);
  const [sessionSecs, setSessionSecs] = useState(0);

  const distractedSinceRef = useRef<number | null>(null);
  const lastScoreTimeRef = useRef<number>(Date.now());
  const sessionStartRef = useRef<number>(Date.now());
  const focusedStreakRef = useRef<number>(0);

  const updateEngine = (signals: { head_distracted: boolean, gaze_distracted: boolean, drowsy: boolean, distractor_count: number }) => {
    const now = Date.now();
    const any_distracted = signals.head_distracted || signals.gaze_distracted || signals.drowsy || signals.distractor_count > 0;
    
    setIsDistracted(any_distracted);

    let currentDistractionSecs = 0;
    if (any_distracted) {
      if (distractedSinceRef.current === null) {
        distractedSinceRef.current = now;
      }
      currentDistractionSecs = (now - distractedSinceRef.current) / 1000.0;
      focusedStreakRef.current = 0.0;
    } else {
      if (distractedSinceRef.current !== null) {
        // Distraction ended
        distractedSinceRef.current = null;
      }
      currentDistractionSecs = 0.0;
      focusedStreakRef.current += (now - lastScoreTimeRef.current) / 1000.0;
    }

    setDistractionSecs(currentDistractionSecs);
    setFocusedStreak(focusedStreakRef.current);
    setSessionSecs((now - sessionStartRef.current) / 1000.0);

    // Compute nudge level
    let currentNudgeLevel = 0;
    const NUDGE_LEVELS = [
      { secs: 30, level: 4 },
      { secs: 20, level: 3 },
      { secs: 10, level: 2 },
      { secs: 5,  level: 1 },
    ];
    for (const { secs, level } of NUDGE_LEVELS) {
      if (currentDistractionSecs >= secs) {
        currentNudgeLevel = level;
        break;
      }
    }
    setNudgeLevel(currentNudgeLevel);

    // Update focus score every second
    const elapsed = (now - lastScoreTimeRef.current) / 1000.0;
    if (elapsed >= 1.0) {
      setScore(prevScore => {
        let newScore = prevScore;
        if (any_distracted) {
          const decay = 2.0 + (currentDistractionSecs / 10.0);
          newScore = Math.max(0.0, prevScore - decay * elapsed);
        } else {
          newScore = Math.min(100.0, prevScore + 1.5 * elapsed);
        }
        return newScore;
      });
      lastScoreTimeRef.current = now;
    }
  };

  return { score, nudgeLevel, isDistracted, distractionSecs, focusedStreak, sessionSecs, updateEngine };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [cameraStatus, setCameraStatus] = useState("Initializing...");
  const [stats, setStats] = useState<any>(null);

  const engine = useFocusEngine();

  // Handle Distraction Audio Alert
  useEffect(() => {
    if (engine.isDistracted && engine.distractionSecs >= 5) {
      if (audioRef.current && audioRef.current.paused) {
        // Play sound if not already playing
        audioRef.current.play().catch(e => console.error("Audio autoplay blocked:", e));
      }
    } else {
      if (audioRef.current && !audioRef.current.paused) {
        // Stop sound and reset when focused again
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    }
  }, [engine.isDistracted, engine.distractionSecs]);

  // Nudge Messages
  const LNudges: Record<number, string> = {
    0: "✓ On track",
    1: "Stay focused!",
    2: "Hey, you drifted 👀",
    3: "Come back to your work!",
    4: "Take a deep breath. Refocus."
  };

  const nudgeStyles: Record<number, { bg: string, color: string }> = {
    0: { bg: "#1a3a1a", color: "#4ade80" },
    1: { bg: "#3a2a00", color: "#fbbf24" },
    2: { bg: "#3a1a00", color: "#fb923c" },
    3: { bg: "#3a0000", color: "#f87171" },
    4: { bg: "#2a0030", color: "#e879f9" },
  };

  const getScoreColor = (sc: number) => {
    if (sc >= 70) return "#4ade80";
    if (sc >= 40) return "#fb923c";
    return "#f87171";
  };

  useEffect(() => {
    async function setupCamera() {
      console.log("Setting up camera...");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            console.log("Camera metadata loaded");
            setCameraStatus("Running");
            startInference();
          };
        }
      } catch (err) {
        console.error("Camera access error:", err);
        setCameraStatus("CAMERA ERROR");
      }
    }
    setupCamera();

    let isProcessing = false;
    let intervalId: any;

    function startInference() {
       intervalId = setInterval(async () => {
        if (isProcessing) return;
        if (videoRef.current && canvasRef.current) {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video.videoWidth === 0) return;

          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(video, 0, 0);
            isProcessing = true;
            canvas.toBlob(async (blob) => {
              if (blob) {
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");
                try {
                  const API_URL = process.env.NEXT_PUBLIC_API_URL || "";
                  const res = await fetch(`${API_URL}/api/process`, {
                    method: "POST",
                    body: formData,
                  });
                  if (!res.ok) throw new Error("API failed");
                  const data = await res.json();
                  setStats(data);

                  const signals = {
                    head_distracted: data.head_distracted,
                    gaze_distracted: data.gaze_distracted,
                    drowsy: data.drowsy,
                    distractor_count: data.distractors ? data.distractors.length : 0
                  };
                  engine.updateEngine(signals);
                  
                } catch (e) {
                  console.error("Inference error:", e);
                  // Keep silent here rather than breaking UI mostly, but setting state helps
                  // setCameraStatus("BACKEND ERROR");
                } finally {
                  isProcessing = false;
                }
              } else {
                isProcessing = false;
              }
            }, "image/jpeg", 0.9);
          }
        }
      }, 100);
    }

    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); 

  const NudgeComponent = ({ level }: { level: number }) => {
    if (engine.isDistracted) {
      const msg = level > 0 ? LNudges[level] : "Distracted";
      const bg = level > 0 ? nudgeStyles[level].bg : "#3a0000";
      const color = level > 0 ? nudgeStyles[level].color : "#f87171";
      return (
        <div style={{
          padding: "12px 20px",
          borderRadius: "10px",
          textAlign: "center",
          fontSize: "18px",
          fontWeight: 600,
          margin: "8px 0",
          backgroundColor: bg,
          color: color
        }}>
          {msg}
        </div>
      );
    }
    return (
      <div style={{
        padding: "12px 20px",
        borderRadius: "10px",
        textAlign: "center",
        fontSize: "18px",
        fontWeight: 600,
        margin: "8px 0",
        backgroundColor: nudgeStyles[0].bg,
        color: nudgeStyles[0].color
      }}>
        {LNudges[0]}
      </div>
    );
  };

  const Signal = ({ ok, label }: { ok: boolean, label: string }) => (
    <span style={{ color: ok ? "#4ade80" : "#f87171", fontWeight: 600, marginRight: "12px" }}>
      {ok ? "✓" : "✗"} {label}
    </span>
  );

  return (
    <main style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto", fontFamily: "sans-serif" }}>
      <h1 style={{ marginBottom: "0.5rem" }}>Focus Guardian 🎯</h1>
      <p style={{ color: "#aaa", marginBottom: "2rem" }}>Privacy-first distraction detection. Everything runs locally.</p>
      
      {/* Hidden Audio Element for Alerts */}
      <audio ref={audioRef} src="/sound.mp4" loop />
      
      {!stats && cameraStatus !== "Running" && (
         <div style={{ padding: "1rem", backgroundColor: "#1e293b", borderRadius: "8px", marginBottom: "1rem" }}>
           <p>{cameraStatus}</p>
         </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "2rem" }}>
        {/* Camera Side */}
        <div>
          <h3 style={{ marginBottom: "1rem" }}>Live Feed</h3>
          <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", borderRadius: "12px", background: "#000", border: engine.isDistracted ? "2px solid #f87171" : "2px solid #334155" }} />
          <canvas ref={canvasRef} style={{ display: "none" }} />
          
          <div style={{ marginTop: "1rem" }}>
             <NudgeComponent level={engine.nudgeLevel} />
          </div>
        </div>

        {/* Stats Side */}
        <div>
          <h3 style={{ marginBottom: "1rem" }}>Focus Score</h3>
          <div style={{ background: "#1e1e2e", borderRadius: "10px", padding: "2rem 1rem", textAlign: "center", marginBottom: "1rem" }}>
            <div style={{ fontSize: "72px", fontWeight: 700, lineHeight: 1, color: getScoreColor(engine.score) }}>
              {Math.round(engine.score)}
            </div>
            <p style={{ color: "#888", marginTop: "0.5rem" }}>/ 100</p>
          </div>

          <div style={{ background: "#1e1e2e", borderRadius: "10px", padding: "1.5rem", lineHeight: "1.8" }}>
            <div style={{ marginBottom: "1rem", display: "flex", flexWrap: "wrap", gap: "8px" }}>
              <Signal ok={!stats?.head_distracted} label="Head straight" />
              <Signal ok={!stats?.gaze_distracted} label="Gaze on screen" />
              <Signal ok={!stats?.drowsy} label="Awake" />
              <Signal ok={!(stats?.distractors?.length > 0)} label="No distractors" />
            </div>

            <div style={{ color: "#cbd5e1", marginTop: "1rem", borderTop: "1px solid #334155", paddingTop: "1rem" }}>
              <p><b>Session:</b> {Math.floor(engine.sessionSecs / 60).toString().padStart(2, '0')}:{Math.floor(engine.sessionSecs % 60).toString().padStart(2, '0')}</p>
              <p><b>Streak:</b> {Math.round(engine.focusedStreak)}s</p>
            </div>
            
            {stats && (
              <div style={{ color: "#94a3b8", fontSize: "0.9rem", marginTop: "1rem" }}>
                <p>Yaw: {stats.yaw}° | Pitch: {stats.pitch}°</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
