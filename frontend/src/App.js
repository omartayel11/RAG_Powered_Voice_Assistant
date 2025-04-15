import React, { useState } from "react";
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const res = await axios.post("http://localhost:8001/chat", {
        message: input,
      });
      const botMessage = { sender: "bot", text: res.data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        sender: "bot",
        text: "حدث خطأ أثناء الاتصال بالخادم.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }

    setInput("");
  };

  return (
    <div style={{ maxWidth: 600, margin: "0 auto", padding: 20 }}>
      <h2 style={{ textAlign: "center" }}>🤖 مساعد الوصفات</h2>

      <div
        style={{
          height: 400,
          overflowY: "auto",
          border: "1px solid #ccc",
          padding: 10,
          marginBottom: 10,
          borderRadius: 8,
          backgroundColor: "#f9f9f9",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              textAlign: msg.sender === "user" ? "right" : "left",
              backgroundColor: msg.sender === "user" ? "#d1e7dd" : "#f8d7da",
              padding: "10px",
              margin: "6px 0",
              borderRadius: "10px",
              maxWidth: "80%",
              marginLeft: msg.sender === "user" ? "auto" : 0,
              marginRight: msg.sender === "user" ? 0 : "auto",
            }}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <form onSubmit={sendMessage} style={{ display: "flex", gap: 10 }}>
        <input
          type="text"
          placeholder="اكتب سؤالك هنا..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          style={{ flex: 1, padding: 10 }}
        />
        <button type="submit" style={{ padding: "0 20px" }}>
          إرسال
        </button>
      </form>
    </div>
  );
}

export default App;
