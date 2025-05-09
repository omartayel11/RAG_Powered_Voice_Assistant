import React, { useState, useEffect, useRef } from "react";

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [expectingChoice, setExpectingChoice] = useState(false);
  const [ws, setWs] = useState(null);
  const [currentRecipeTitle, setCurrentRecipeTitle] = useState(null);
  const [favourites, setFavourites] = useState([]);
  const [fullRecipeContent, setFullRecipeContent] = useState({});
  const messageListRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
  }, []);

  const connectWebSocket = () => {
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
        
        if (data.selected_title && data.full_recipe) {
          setCurrentRecipeTitle(data.selected_title);
          setFullRecipeContent(prev => ({
            ...prev,
            [data.selected_title]: data.full_recipe
          }));
        }
      }
      
       else if (data.type === "error") {
        setMessages((prev) => [...prev, { sender: "bot", text: data.message }]);
      }
    };

    setWs(socket);
  };

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
    const selectedTitle = suggestions[index];
    const selected = `${index + 1}. ${selectedTitle}`;
    setMessages((prev) => [...prev, { sender: "user", text: selected }]);
    ws.send(String(index + 1));
    setExpectingChoice(false);
    setSuggestions([]);
    setCurrentRecipeTitle(selectedTitle); // Set selected recipe title
  };

  const handleAddToFavourites = async () => {
    if (!currentRecipeTitle || !ws) return;
  
    const email = localStorage.getItem("userEmail");
    if (!email) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❗ عذرًا، يجب تسجيل الدخول أولاً لإضافة إلى المفضلة.` },
      ]);
      return;
    }
  
    try {
      const response = await fetch("http://localhost:8001/add-favourite", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email,
          title: currentRecipeTitle,
          recipe: fullRecipeContent[currentRecipeTitle] || "", // 💡 We'll prepare this shortly
        }),
      });
  
      const result = await response.json();
  
      if (result.status === "success") {
        setFavourites((prev) => [...prev, currentRecipeTitle]);
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: `✅ تمت إضافة الوصفة "${currentRecipeTitle}" إلى المفضلة.` },
        ]);
      } else if (result.status === "exists") {
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: `🔔 الوصفة "${currentRecipeTitle}" موجودة بالفعل في المفضلة.` },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: `❗ حدث خطأ أثناء إضافة الوصفة.` },
        ]);
      }
    } catch (error) {
      console.error("Error adding to favourites:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❗ حدث خطأ أثناء الاتصال بالخادم.` },
      ]);
    }
  
    setCurrentRecipeTitle(null);
  };
  

  const handleNewChat = () => {
    if (ws) {
      ws.close();
      setWs(null);
    }

    setMessages([]);
    setInput("");
    setSuggestions([]);
    setExpectingChoice(false);
    setCurrentRecipeTitle(null);

    setTimeout(() => {
      connectWebSocket();
    }, 200);
  };

  return (
    <div className="App">
      <h1>Food Chat</h1>

      <button style={{ marginBottom: "10px" }} onClick={handleNewChat}>
        Start New Chat
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

      {/* Add to Favourites button */}
      {currentRecipeTitle && (
        <div style={{ marginTop: "1rem", textAlign: "center" }}>
          <p>📖 الوصفة الحالية: <strong>{currentRecipeTitle}</strong></p>
          <button
            onClick={handleAddToFavourites}
            style={{
              marginTop: "8px",
              padding: "10px 20px",
              backgroundColor: "#f9a825",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            ➕ إضافة إلى المفضلة
          </button>
        </div>
      )}

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
