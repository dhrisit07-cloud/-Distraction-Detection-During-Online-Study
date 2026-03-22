"use client";

import { useRef, useEffect, useState } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState("Initializing...");
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    async function setupCamera() {
      console.log("Setting up camera...");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            console.log("Camera metadata loaded:", videoRef.current?.videoWidth, "x", videoRef.current?.videoHeight);
            startInference();
          };
        }
      } catch (err) {
        console.error("Camera access error:", err);
        setStatus("CAMERA ERROR");
      }
    }
    setupCamera();

    let intervalId: any;
    function startInference() {
       intervalId = setInterval(async () => {
        if (videoRef.current && canvasRef.current) {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video.videoWidth === 0) return;

          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(video, 0, 0);
            canvas.toBlob(async (blob) => {
              if (blob) {
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");
                try {
                  const res = await fetch("/api/process", {
                    method: "POST",
                    body: formData,
                  });
                  if (!res.ok) throw new Error("API failed");
                  const data = await res.json();
                  console.log("Inference result:", data);
                  setStats(data);
                  if ((data.distractors && data.distractors.length > 0) || data.drowsy || data.head_distracted || data.gaze_distracted) {
                    setStatus("DISTRACTION DETECTED");
                  } else {
                    setStatus("FOCUSED");
                  }
                } catch (e) {
                  console.error("Inference error:", e);
                  setStatus("BACKEND ERROR");
                }
              }
            }, "image/jpeg", 0.5);
          }
        }
      }, 500); // 2 fps for reliable debugging
    }

    return () => clearInterval(intervalId);
  }, []);

  return (
    <main style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      <h1>Focus Guardian 🎯</h1>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
        <div>
          <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", borderRadius: "12px", background: "#000" }} />
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>
        <div style={{ background: "#1e293b", padding: "1.5rem", borderRadius: "12px" }}>
          <h2 style={{ color: status === "FOCUSED" ? "#4ade80" : "#f87171" }}>{status}</h2>
          {stats && (
            <div style={{ marginTop: "1rem", fontSize: "1.1rem" }}>
              <p>Yaw: {stats.yaw}°</p>
              <p>Pitch: {stats.pitch}°</p>
              <p>Drowsy: {stats.drowsy ? "YES" : "NO"}</p>
              <p style={{ marginTop: "10px", color: stats.distractors && stats.distractors.length > 0 ? "#f87171" : "#fff"  }}>
                Distractors: {stats.distractors && stats.distractors.length > 0 ? stats.distractors.join(", ") : "None"}
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
