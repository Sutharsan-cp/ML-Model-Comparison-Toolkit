import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Brain, 
  TrendingUp, 
  BarChart3, 
  Zap, 
  Sparkles, 
  Github,
  Play
} from 'lucide-react';
import './Home.css';

const Home = () => {
  const features = [
    {
      icon: <Brain size={48} />,
      title: "Multiple Models",
      description: "Compare various ML algorithms from Linear Regression to XGBoost"
    },
    {
      icon: <BarChart3 size={48} />,
      title: "Visual Analytics",
      description: "Beautiful charts and metrics to understand model performance"
    },
    {
      icon: <Zap size={48} />,
      title: "XAI Integration",
      description: "SHAP and LIME explanations for model interpretability"
    },
    {
      icon: <Sparkles size={48} />,
      title: "Smart Recommendations",
      description: "Get AI-powered model recommendations for your dataset"
    }
  ];

  const stats = [
    { value: "15+", label: "ML Models" },
    { value: "99%", label: "Accuracy" },
    { value: "24/7", label: "Available" },
    { value: "100%", label: "Open Source" }
  ];

  return (
    <div className="homepage">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="floating-shapes">
            <div className="shape shape-1"></div>
            <div className="shape shape-2"></div>
            <div className="shape shape-3"></div>
          </div>
        </div>
        
        <div className="hero-content">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="hero-text"
          >
            <h1 className="hero-title">
              ML Model
              <span className="gradient-text"> Comparison</span>
              Toolkit
            </h1>
            <p className="hero-description">
              Compare, analyze, and interpret machine learning models with beautiful visualizations 
              and explainable AI. Your all-in-one platform for model evaluation.
            </p>
            
            <div className="hero-buttons">
              <Link to="/dashboard" className="cta-button primary">
                <Play size={20} />
                Get Started
              </Link>
              <a href="#features" className="cta-button secondary">
                Learn More
              </a>
            </div>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="hero-visual"
          >
            <div className="ml-visual">
              <div className="model-card visual-card">
                <div className="card-header">
                  <div className="card-dots">
                    <div className="dot red"></div>
                    <div className="dot yellow"></div>
                    <div className="dot green"></div>
                  </div>
                </div>
                <div className="chart-placeholder">
                  <div className="metric-bars">
                    {[80, 90, 85, 95, 88].map((height, index) => (
                      <div 
                        key={index}
                        className="metric-bar"
                        style={{ height: `${height}%` }}
                      ></div>
                    ))}
                  </div>
                </div>
                <div className="model-tags">
                  <span className="tag">XGBoost</span>
                  <span className="tag">95% ACC</span>
                </div>
              </div>
              
              <div className="floating-element element-1">
                <TrendingUp size={24} />
              </div>
              <div className="floating-element element-2">
                <BarChart3 size={24} />
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="stat-card"
            >
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="section-header"
        >
          <h2 className="section-title">Powerful Features</h2>
          <p className="section-description">
            Everything you need to compare and evaluate machine learning models
          </p>
        </motion.div>

        <div className="features-grid">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -5 }}
              className="feature-card"
            >
              <div className="feature-icon">
                {feature.icon}
              </div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="cta-card"
        >
          <h2 className="cta-title">Ready to Compare Models?</h2>
          <p className="cta-description">
            Start exploring your models with our powerful comparison toolkit
          </p>
          <Link to="/dashboard" className="cta-button large">
            <Play size={20} />
            Launch Dashboard
          </Link>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <Brain size={32} className="footer-logo" />
            <span className="footer-title">ML Comparison Toolkit</span>
          </div>
          <div className="footer-links">
            <a href="https://github.com" className="footer-link">
              <Github size={20} />
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;