import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronDown, 
  ChevronUp, 
  Play,
  BarChart3,
  TrendingUp,
  Target,
  Brain,
  Zap,
  Sparkles
} from 'lucide-react';
import axios from 'axios';
import './ModelComparison.css';

const ModelComparison = ({ 
  selectedDataset,
  onTrainingComplete 
}) => {
  const [availableModels, setAvailableModels] = useState({});
  const [selectedModels, setSelectedModels] = useState([]);
  const [expandedCategory, setExpandedCategory] = useState('supervised');
  const [comparisonResults, setComparisonResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [taskType, setTaskType] = useState('classification');

  useEffect(() => {
    loadAvailableModels();
  }, [taskType]);

  const loadAvailableModels = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/api/available-models?task_type=${taskType}`);
      
      // Organize models by category
      const modelsByCategory = {
        supervised: {
          classification: response.data.models.filter(model => 
            model.includes('logistic') || model.includes('xgboost')
          ),
          regression: response.data.models.filter(model => 
            model.includes('linear') || model.includes('regression')
          )
        },
        unsupervised: {
          clustering: ['K-Means', 'DBSCAN', 'Hierarchical'],
          dimensionality: ['PCA', 't-SNE', 'UMAP']
        }
      };
      
      setAvailableModels(modelsByCategory);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const toggleModelSelection = (model) => {
    const updatedModels = selectedModels.includes(model)
      ? selectedModels.filter(m => m !== model)
      : [...selectedModels, model];
    setSelectedModels(updatedModels);
  };

  const trainModels = async () => {
    if (!selectedDataset || selectedModels.length === 0) return;

    setIsTraining(true);
    setComparisonResults(null);

    try {
      const response = await axios.post('http://localhost:8000/api/train-multiple', {
        models: selectedModels,
        dataset_path: selectedDataset.path,
        target_column: selectedDataset.targetColumn,
        task_type: taskType,
        parameters: {}
      });

      if (response.data.status === 'completed') {
        setComparisonResults(response.data.results);
        if (onTrainingComplete) {
          onTrainingComplete(response.data.results);
        }
      }
    } catch (error) {
      console.error('Error training models:', error);
      alert('Error training models. Please check the console for details.');
    } finally {
      setIsTraining(false);
    }
  };

  const getModelIcon = (modelName) => {
    if (modelName.includes('xgboost')) return <Zap size={16} />;
    if (modelName.includes('logistic')) return <Brain size={16} />;
    if (modelName.includes('linear')) return <TrendingUp size={16} />;
    return <Brain size={16} />;
  };

  return (
    <div className="model-comparison">
      {/* Task Type Selection */}
      <div className="task-type-selector">
        <h3>Select Task Type</h3>
        <div className="task-buttons">
          <button 
            className={`task-button ${taskType === 'classification' ? 'active' : ''}`}
            onClick={() => setTaskType('classification')}
          >
            <Target size={20} />
            Classification
          </button>
          <button 
            className={`task-button ${taskType === 'regression' ? 'active' : ''}`}
            onClick={() => setTaskType('regression')}
          >
            <TrendingUp size={20} />
            Regression
          </button>
        </div>
      </div>

      {/* Model Selection Panel */}
      <div className="selection-panel">
        <h2>Select Models to Compare</h2>
        
        {Object.entries(availableModels).map(([category, subcategories]) => (
          <div key={category} className="model-category">
            <button 
              className="category-header"
              onClick={() => setExpandedCategory(expandedCategory === category ? null : category)}
            >
              <span className="category-title">
                {category.charAt(0).toUpperCase() + category.slice(1)} Learning
              </span>
              {expandedCategory === category ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
            </button>
            
            <AnimatePresence>
              {expandedCategory === category && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="subcategories"
                >
                  {Object.entries(subcategories).map(([type, models]) => (
                    <div key={type} className="subcategory">
                      <h4>{type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                      <div className="models-grid">
                        {models.map(model => (
                          <label key={model} className="model-checkbox">
                            <input
                              type="checkbox"
                              checked={selectedModels.includes(model)}
                              onChange={() => toggleModelSelection(model)}
                            />
                            <span className="checkmark"></span>
                            <span className="model-icon">
                              {getModelIcon(model)}
                            </span>
                            {model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
        
        <button 
          className={`train-button ${isTraining ? 'training' : ''}`}
          onClick={trainModels}
          disabled={!selectedDataset || selectedModels.length === 0 || isTraining}
        >
          {isTraining ? (
            <div className="spinner"></div>
          ) : (
            <Play size={16} />
          )}
          {isTraining ? 'Training...' : `Compare Selected Models (${selectedModels.length})`}
        </button>

        {/* Dataset Info */}
        {selectedDataset && (
          <div className="dataset-info">
            <h4>Selected Dataset</h4>
            <p><strong>Name:</strong> {selectedDataset.name}</p>
            <p><strong>Target:</strong> {selectedDataset.targetColumn}</p>
            <p><strong>Samples:</strong> {selectedDataset.samples}</p>
          </div>
        )}
      </div>

      {/* Results Panel */}
      <div className="results-panel">
        {comparisonResults ? (
          <div className="comparison-results">
            <h2>Model Comparison Results</h2>
            
            {/* Metrics Comparison */}
            <div className="metrics-grid">
              {Object.entries(comparisonResults).map(([modelName, result], index) => (
                result.status === 'success' ? (
                  <motion.div
                    key={modelName}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="model-metric-card"
                  >
                    <div className="model-header">
                      <span className="model-icon">
                        {getModelIcon(modelName)}
                      </span>
                      <h3 className="model-name">
                        {modelName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </h3>
                    </div>
                    
                    <div className="metrics-list">
                      {Object.entries(result.metrics).map(([metric, value]) => (
                        <div key={metric} className="metric-row">
                          <div className="metric">
                            <span>{metric.replace(/_/g, ' ').toUpperCase()}</span>
                          </div>
                          <div className="metric-value">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* XAI Features */}
                    {result.xai_data && result.xai_data.feature_importance && (
                      <div className="xai-section">
                        <h4>Feature Importance</h4>
                        <div className="feature-importance">
                          {Object.entries(result.xai_data.feature_importance)
                            .slice(0, 3)
                            .map(([feature, importance]) => (
                              <div key={feature} className="feature-bar">
                                <span className="feature-name">{feature}</span>
                                <div className="bar-container">
                                  <div 
                                    className="bar-fill"
                                    style={{ width: `${importance * 100}%` }}
                                  ></div>
                                </div>
                                <span className="importance-value">
                                  {(importance * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))
                          }
                        </div>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key={modelName}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="model-metric-card error"
                  >
                    <h3 className="model-name">{modelName}</h3>
                    <div className="error-message">
                      {result.message || 'Training failed'}
                    </div>
                  </motion.div>
                )
              ))}
            </div>

            {/* Charts Section */}
            <div className="charts-section">
              <div className="chart-card">
                <h3>Accuracy Comparison</h3>
                <div className="bar-chart">
                  {Object.entries(comparisonResults)
                    .filter(([_, result]) => result.status === 'success')
                    .map(([modelName, result]) => (
                      <div key={modelName} className="bar-container">
                        <div 
                          className="bar" 
                          style={{ 
                            height: `${result.metrics.accuracy * 100}%`,
                            background: `linear-gradient(to top, #ff6b6b, #ffd93d)`
                          }}
                        ></div>
                        <span className="bar-label">
                          {modelName.split('_')[0]}
                        </span>
                        <span className="bar-value">
                          {(result.metrics.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))
                  }
                </div>
              </div>

              <div className="chart-card">
                <h3>Metrics Overview</h3>
                <div className="metrics-radar">
                  {Object.entries(comparisonResults)
                    .filter(([_, result]) => result.status === 'success')
                    .map(([modelName, result]) => (
                      <div key={modelName} className="model-metrics">
                        <h4>{modelName.split('_')[0]}</h4>
                        <div className="metric-dots">
                          {Object.entries(result.metrics).map(([metric, value]) => (
                            <div 
                              key={metric}
                              className="metric-dot"
                              style={{ 
                                '--size': `${value * 100}%`,
                                '--hue': Object.keys(result.metrics).indexOf(metric) * 60
                              }}
                              title={`${metric}: ${value.toFixed(3)}`}
                            ></div>
                          ))}
                        </div>
                      </div>
                    ))
                  }
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <BarChart3 size={64} />
            <h3>No Comparison Results</h3>
            <p>Select models and click "Compare Selected Models" to see results</p>
            {!selectedDataset && (
              <p className="warning">Please select a dataset first</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelComparison;