import React, { useState, useEffect, useRef } from "react";

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [expectingChoice, setExpectingChoice] = useState(false);
  const [ws, setWs] = useState(null);
  const messageListRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8001/ws/chat");

    socket.onopen = () => {
      console.log("🟢 WebSocket connected.");
      const email = localStorage.getItem("userEmail");
      socket.send(JSON.stringify({ email }));
    };

    socket.onclose = () => console.log("🔴 WebSocket disconnected.");
    socket.onerror = (error) => console.error("WebSocket error:", error);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "suggestions") {
        setSuggestions(data.suggestions);
        setExpectingChoice(true);
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: data.message || "اختر وصفة من الخيارات التالية:" },
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
    setMessages((prev) => [...prev, { sender: "user", text: input }]);
    ws.send(input);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  const handleSuggestionClick = (index) => {
    if (!ws) return;
    const selected = `${index + 1}. ${suggestions[index]}`;
    setMessages((prev) => [...prev, { sender: "user", text: selected }]);
    ws.send(String(index + 1));
    setExpectingChoice(false);
    setSuggestions([]);
  };

  return (
    <div className="App">
      <h1>🍽️ روبوت الوصفات العربي</h1>
      <button
  style={{ marginBottom: "10px" }}
  onClick={() => {
    if (ws) {
      ws.send("/new");
      setMessages((prev) => [...prev, { sender: "user", text: "🆕 بدء محادثة جديدة" }]);
      setSuggestions([]);
      setExpectingChoice(false);
    }
  }}
>
  بدء محادثة جديدة
</button>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}

        {expectingChoice && (
          <>
            <div className="choice-hint">⬇️ من فضلك اختر وصفة 👇</div>
            <div className="suggestions">
              {suggestions.map((s, i) => (
                <button key={i} onClick={() => handleSuggestionClick(i)}>
                  {i + 1}. {s}
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

export default Chat;
