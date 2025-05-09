import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function Signup() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
    password: "",
    name: "",
    gender: "male",
    profession: "",
    allergies: "",
    likes: "",
    dislikes: ""
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSignup = async () => {
    const payload = {
      ...form,
      allergies: form.allergies.split(",").map((a) => a.trim()),
      likes: form.likes.split(",").map((l) => l.trim()),
      dislikes: form.dislikes.split(",").map((d) => d.trim()),
    };

    const res = await fetch("http://localhost:8001/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      alert("Successfuly registered!!!");
      navigate("/login");
    } else {
      const err = await res.json();
      alert("Unable to Signup" + err.detail);
    }
  };

  return (
    <div className="App">
      <h2>Signup</h2>
      <div className="form">
        <input name="email" type="email" placeholder="Email" onChange={handleChange} />
        <input name="password" type="password" placeholder="Password" onChange={handleChange} />
        <input name="name" placeholder="First Name (optional)" onChange={handleChange} />
        <select name="gender" value={form.gender} onChange={handleChange}>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
        <input name="profession" placeholder="Profession (optional)" onChange={handleChange} />
        <input name="likes" placeholder="Likes" onChange={handleChange} />
        <input name="dislikes" placeholder="Dislikes" onChange={handleChange} />
        <input name="allergies" placeholder="Allergies (EX: Onions, Milk)" onChange={handleChange} />
        <button onClick={handleSignup}>signup</button>
      </div>
    </div>
  );
}

export default Signup;
