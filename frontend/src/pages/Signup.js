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
      alert("تم التسجيل بنجاح! سجل الدخول الآن.");
      navigate("/login");
    } else {
      const err = await res.json();
      alert("فشل التسجيل: " + err.detail);
    }
  };

  return (
    <div className="App">
      <h2>📋 إنشاء حساب جديد</h2>
      <div className="form">
        <input name="email" type="email" placeholder="البريد الإلكتروني" onChange={handleChange} />
        <input name="password" type="password" placeholder="كلمة المرور" onChange={handleChange} />
        <input name="name" placeholder="الاسم (اختياري)" onChange={handleChange} />
        <select name="gender" value={form.gender} onChange={handleChange}>
          <option value="male">ذكر</option>
          <option value="female">أنثى</option>
        </select>
        <input name="profession" placeholder="المهنة (اختياري)" onChange={handleChange} />
        <input name="likes" placeholder="الأكلات المفضلة (مفصولة بفاصلة)" onChange={handleChange} />
        <input name="dislikes" placeholder="الأكلات غير المفضلة" onChange={handleChange} />
        <input name="allergies" placeholder="الحساسيات (مثلاً: بصل، لبن)" onChange={handleChange} />
        <button onClick={handleSignup}>تسجيل</button>
      </div>
    </div>
  );
}

export default Signup;
