import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const iconPaths = {
  audio: "M9 18V5l12-2v13 M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm12-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z",
  file: "M6 2h8l4 4v16H6V2Zm8 0v5h5 M9 13h6 M9 17h6",
  logout: "M15 3h4v18h-4 M10 17l5-5-5-5 M15 12H3",
  play: "M8 5v14l11-7-11-7Z",
  send: "M21 3 10 14 M21 3l-7 18-4-7-7-4 18-7Z",
  upload: "M12 16V4 M7 9l5-5 5 5 M5 20h14",
  video: "M4 6h11v12H4V6Zm11 4 5-3v10l-5-3",
};

function Icon({ name, size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d={iconPaths[name]} />
    </svg>
  );
}

function App() {
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [authMode, setAuthMode] = useState("login");
  const [email, setEmail] = useState("demo@example.com");
  const [password, setPassword] = useState("password123");
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [timestamps, setTimestamps] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const mediaRef = useRef(null);

  const headers = useMemo(() => ({ Authorization: `Bearer ${token}` }), [token]);

  async function request(path, options = {}) {
    const response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: { ...(options.headers || {}), ...(token ? headers : {}) },
    });
    if (response.status === 401) {
      signOut("Your session expired. Please log in again.");
      throw new Error("Your session expired. Please log in again.");
    }
    if (!response.ok) throw new Error((await response.json()).detail || "Request failed");
    return response.json();
  }

  async function authenticate(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const data = await request(`/auth/${authMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      localStorage.setItem("token", data.access_token);
      setToken(data.access_token);
      await loadFiles(data.access_token);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadFiles(nextToken = token) {
    const response = await fetch(`${API_URL}/files`, { headers: { Authorization: `Bearer ${nextToken}` } });
    if (response.ok) {
      const data = await response.json();
      setFiles(data);
      setSelected(data[0] || null);
    }
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    setError("");
    setBusy(true);
    try {
      const uploaded = await request("/upload", { method: "POST", body });
      const nextFiles = [uploaded, ...files.filter((item) => item.id !== uploaded.id)];
      setFiles(nextFiles);
      setSelected(uploaded);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function ask(event) {
    event.preventDefault();
    if (!question.trim()) return;
    setError("");
    const userMessage = { role: "user", content: question };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setBusy(true);
    try {
      const data = await request("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, file_id: selected?.id || null }),
      });
      setMessages((current) => [...current, { role: "assistant", content: data.answer, sources: data.sources }]);
      setTimestamps(data.sources.filter((source) => source.start_time !== null));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function searchTopic(topic) {
    if (!topic.trim()) return;
    setError("");
    try {
      const data = await request("/timestamps/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, file_id: selected?.id || null }),
      });
      setTimestamps(data);
    } catch (err) {
      setError(err.message);
    }
  }

  function playAt(seconds) {
    if (mediaRef.current) {
      mediaRef.current.currentTime = seconds;
      mediaRef.current.play();
    }
  }

  function signOut(message = "") {
    localStorage.removeItem("token");
    setToken("");
    setFiles([]);
    setSelected(null);
    setMessages([]);
    setTimestamps([]);
    setError(message);
  }

  if (!token) {
    return (
      <main className="auth-shell">
        <section className="auth-panel">
          <div>
            <p className="eyebrow">AI Doc Q&A</p>
            <h1>Ask every file what matters.</h1>
          </div>
          <form onSubmit={authenticate} className="auth-form">
            <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="Email" />
            <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="Password" />
            <button disabled={busy}>{authMode === "login" ? "Log in" : "Create account"}</button>
          </form>
          {error && <p className="error-text">{error}</p>}
          <button className="link-button" onClick={() => setAuthMode(authMode === "login" ? "signup" : "login")}>
            {authMode === "login" ? "Need an account?" : "Already have an account?"}
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <h1>AI Doc Q&A</h1>
          <button className="icon-button" onClick={signOut} title="Log out"><Icon name="logout" size={18} /></button>
        </div>
        <label className="upload-target">
          <Icon name="upload" size={26} />
          <span>Upload PDF, audio, or video</span>
          <input type="file" accept=".pdf,.txt,audio/*,video/*" onChange={upload} />
        </label>
        {error && <p className="sidebar-error">{error}</p>}
        <div className="file-list">
          {files.map((file) => (
            <button key={file.id} className={selected?.id === file.id ? "file-card active" : "file-card"} onClick={() => setSelected(file)}>
              {file.kind === "video" ? <Icon name="video" /> : file.kind === "audio" ? <Icon name="audio" /> : <Icon name="file" />}
              <span>{file.filename}</span>
            </button>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <section className="summary-band">
          <div>
            <p className="eyebrow">{selected ? selected.kind : "No file selected"}</p>
            <h2>{selected?.filename || "Upload a file to begin"}</h2>
            <p>{selected?.summary || "Summaries and grounded answers will appear after processing."}</p>
          </div>
          {selected?.kind === "video" && (
            <video ref={mediaRef} controls className="media-player" src={`${API_URL}/files/${selected.id}/media?token=${token}`} />
          )}
          {selected?.kind === "audio" && (
            <audio ref={mediaRef} controls className="media-player" src={`${API_URL}/files/${selected.id}/media?token=${token}`} />
          )}
        </section>

        <section className="content-grid">
          <div className="chat-panel">
            <div className="messages">
              {messages.map((message, index) => (
                <article key={index} className={`message ${message.role}`}>
                  <p>{message.content}</p>
                  {message.sources?.length > 0 && (
                    <div className="sources">
                      {message.sources.map((source) => (
                        <span key={source.chunk_id}>
                          {source.page_number ? `Page ${source.page_number}` : `${source.start_time ?? 0}s`}
                        </span>
                      ))}
                    </div>
                  )}
                </article>
              ))}
            </div>
            <form onSubmit={ask} className="chat-form">
              <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about the selected upload" />
              <button disabled={busy || !selected} title="Send"><Icon name="send" size={18} /></button>
            </form>
          </div>

          <aside className="timestamp-panel">
            <h2>Timestamps</h2>
            <button className="topic-button" onClick={() => searchTopic(question || "main topic")} disabled={!selected}>Find Topic</button>
            <div className="timestamp-list">
              {timestamps.map((item) => (
                <article className="timestamp-card" key={item.chunk_id}>
                  <p>{item.text}</p>
                  <button onClick={() => playAt(item.start_time || 0)}><Icon name="play" size={15} /> {Math.round(item.start_time || 0)}s</button>
                </article>
              ))}
            </div>
          </aside>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
