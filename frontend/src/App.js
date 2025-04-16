import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]); // full chat history
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [expectingChoice, setExpectingChoice] = useState(false);
  const [ws, setWs] = useState(null);

  const messageListRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8001/ws/chat");
    socket.onopen = () => console.log("🟢 WebSocket connected.");
    socket.onclose = () => console.log("🔴 WebSocket disconnected.");
    socket.onerror = (error) => console.error("WebSocket error:", error);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "suggestions") {
        setSuggestions(data.suggestions);
        setExpectingChoice(true);
        setMessages((prev) => [...prev, { sender: "bot", text: data.message }]);
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
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || !ws) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);

    ws.send(userMessage);
    setInput("");
  };

  const handleSuggestionClick = (index) => {
    if (!ws) return;

    // Add visual confirmation of user's selection
    const selected = `${index + 1}. ${suggestions[index]}`;
    setMessages((prev) => [...prev, { sender: "user", text: selected }]);

    ws.send(String(index));
    setExpectingChoice(false);
    setSuggestions([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="App">
      <h1>🍽️ Arabic Food Chatbot</h1>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}

        {expectingChoice && suggestions.length > 0 && (
          <div className="suggestions">
            {suggestions.map((item, idx) => (
              <button key={idx} onClick={() => handleSuggestionClick(idx)}>
                {idx + 1}. {item}
              </button>
            ))}
          </div>
        )}

        <div ref={messageListRef}></div>
      </div>

      <div className="input-container">
        <input
          type="text"
          placeholder="اكتب رسالتك..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={sendMessage}>إرسال</button>
      </div>
    </div>
  );
}

export default App;
