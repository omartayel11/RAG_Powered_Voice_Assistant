import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    const res = await fetch("http://localhost:8001/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (res.ok) {
      localStorage.setItem("userEmail", email); // persist email
      navigate("/chat");
    } else {
      const err = await res.json();
      alert("فشل تسجيل الدخول: " + err.detail);
    }
  };

  return (
    <div className="App">
      <h2>🔐 تسجيل الدخول</h2>
      <div className="form">
        <input type="email" placeholder="البريد الإلكتروني" value={email} onChange={(e) => setEmail(e.target.value)} />
        <input type="password" placeholder="كلمة المرور" value={password} onChange={(e) => setPassword(e.target.value)} />
        <button onClick={handleLogin}>دخول</button>
      </div>
    </div>
  );
}

export default Login;
