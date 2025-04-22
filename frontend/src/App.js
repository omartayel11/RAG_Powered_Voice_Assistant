import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]); // full chat history
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [expectingChoice, setExpectingChoice] = useState(false);
  const [ws, setWs] = useState(null);

  const messageListRef = useRef(null);

  const [userInfoRequested, setUserInfoRequested] = useState(false);
  const [name, setName] = useState("");
  const [gender, setGender] = useState("male");
  const [profession, setProfession] = useState("");

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8001/ws/chat");

    socket.onopen = () => console.log("🟢 WebSocket connected.");
    socket.onclose = () => console.log("🔴 WebSocket disconnected.");
    socket.onerror = (error) => console.error("WebSocket error:", error);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "collect_user_info") {
        setUserInfoRequested(true);
        return;
      }

      if (data.type === "suggestions") {
        setSuggestions(data.suggestions);
        setExpectingChoice(true);
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: data.message || "اختر وصفة من الخيارات التالية:" }
        ]);
      } else if (data.type === "response") {
        setMessages((prev) => [...prev, { sender: "bot", text: data.message }]);
        setExpectingChoice(false);
        setSuggestions([]);
      } else if (data.type === "error") {
        setMessages((prev) => [...prev, { sender: "bot", text: data.message }]);
      }
    };

    setWs(socket);
    return () => socket.close();
  }, []);

  useEffect(() => {
    messageListRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, suggestions]);

  const sendMessage = () => {
    if (!input.trim() || !ws) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);

    ws.send(userMessage);
    setInput("");
  };

  const submitUserInfo = () => {
    if (!name.trim() || !gender) return;

    const userData = {
      name: name.trim(),
      gender: gender.trim().toLowerCase(),
      profession: profession.trim() || null,
    };

    ws.send(JSON.stringify(userData));
    setUserInfoRequested(false);
  };

  const handleSuggestionClick = (index) => {
    if (!ws) return;

    const selected = `${index + 1}. ${suggestions[index]}`;
    setMessages((prev) => [...prev, { sender: "user", text: selected }]);

    ws.send(String(index + 1)); // send as 1-based index
    setExpectingChoice(false);
    setSuggestions([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  if (userInfoRequested) {
    return (
      <div className="App">
        <h2>👋 أهلاً بيك! عرفنا عليك</h2>
        <div className="form">
          <input
            type="text"
            placeholder="اسمك"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="male">ذكر</option>
            <option value="female">أنثى</option>
          </select>
          <input
            type="text"
            placeholder="المهنة (اختياري)"
            value={profession}
            onChange={(e) => setProfession(e.target.value)}
          />
          <button onClick={submitUserInfo}>ابدأ المحادثة</button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <h1>🍽️ روبوت الوصفات العربي</h1>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}

        {expectingChoice && (
          <>
            <div className="choice-hint">⬇️ من فضلك اختر وصفة من الاختيارات 👇</div>
            <div className="suggestions">
              {suggestions.map((item, idx) => (
                <button key={idx} onClick={() => handleSuggestionClick(idx)}>
                  {idx + 1}. {item}
                </button>
              ))}
            </div>
          </>
        )}

        <div ref={messageListRef}></div>
      </div>

      <div className="input-container">
        <input
          type="text"
          placeholder={expectingChoice ? "بانتظار اختيارك..." : "اكتب رسالتك..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={expectingChoice}
        />
        <button onClick={sendMessage} disabled={expectingChoice}>
          إرسال
        </button>
      </div>
    </div>
  );
}

export default App;
