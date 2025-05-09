import React from "react";
import { Link } from "react-router-dom";
import "./Home.css";

function Home() {
  return (
    <div className="homepage-container">
      <header className="homepage-hero">
        <h1>Recipe Assistant</h1>
        <p className="subtitle">
          Smart, kind, and personalized food help
        </p>
        <div className="homepage-buttons">
          <Link to="/signup">
            <button className="btn primary-btn">Sign Up</button>
          </Link>
          <Link to="/login">
            <button className="btn secondary-btn">Log In</button>
          </Link>
        </div>
      </header>

      {/* <section className="homepage-section">
        <h2>How It Works</h2>
        <p>
          Our assistant talks in Egyptian Arabic, understands your food needs and health
          preferences, and gives you step-by-step guidance to delicious recipes.
        </p>
      </section>

      <section className="homepage-section features">
        <h2>Why Elders Love It</h2>
        <ul>
          <li>🗣️ Easy to use with voice-like interaction</li>
          <li>🌿 Personalized to food habits and sensitivities</li>
          <li>🕒 Recommends meals based on time of day</li>
          <li>😊 Always kind, respectful, and a bit funny</li>
        </ul>
      </section> */}

      <footer className="homepage-footer">
        <p>&copy; 2025 Arabic Recipe Assistant. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default Home;
