import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Database, 
  Play, 
  BarChart3, 
  Settings,
  Download,
  Plus,
  Home,
  Brain
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ModelComparison from '../components/ModelComparison';
import DataUpload from '../components/DataUpload';
import PreprocessingPanel from '../components/PreprocessingPanel';
import styles from '../styles/Dashboard.module.css';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('compare');
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      // Mock API call - replace with actual backend call
      const mockDatasets = [
        { 
          id: 1, 
          name: 'Iris Dataset', 
          type: 'classification', 
          samples: 150,
          path: '/data/processed/classification/iris.csv',
          targetColumn: 'species'
        },
        { 
          id: 2, 
          name: 'Titanic Dataset', 
          type: 'classification', 
          samples: 891,
          path: '/data/processed/classification/titanic.csv',
          targetColumn: 'survived'
        },
        { 
          id: 3, 
          name: 'Boston Housing', 
          type: 'regression', 
          samples: 506,
          path: '/data/processed/regression/boston.csv',
          targetColumn: 'price'
        }
      ];
      setDatasets(mockDatasets);
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  const handleTrainingComplete = (results) => {
    setTrainingResults(results);
    console.log('Training completed:', results);
  };

  const handleHomeClick = () => {
    navigate('/');
  };

  return (
    <div className={styles.dashboard}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <div className={styles.logo} onClick={handleHomeClick}>
            <Brain size={28} />
            <span>ML Toolkit</span>
          </div>
        </div>
        
        <nav className={styles.sidebarNav}>
          <button 
            className={`${styles.navItem} ${activeTab === 'compare' ? styles.active : ''}`}
            onClick={() => setActiveTab('compare')}
          >
            <BarChart3 size={20} />
            <span>Model Comparison</span>
          </button>
          
          <button 
            className={`${styles.navItem} ${activeTab === 'upload' ? styles.active : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={20} />
            <span>Upload Data</span>
          </button>
          
          <button 
            className={`${styles.navItem} ${activeTab === 'preprocess' ? styles.active : ''}`}
            onClick={() => setActiveTab('preprocess')}
          >
            <Settings size={20} />
            <span>Preprocessing</span>
          </button>
        </nav>

        {/* Dataset Selection */}
        <div className={styles.datasetSection}>
          <div className={styles.sectionHeader}>
            <h3>Datasets</h3>
            <button className={styles.addButton}>
              <Plus size={16} />
            </button>
          </div>
          <div className={styles.datasetList}>
            {datasets.map(dataset => (
              <motion.div
                key={dataset.id}
                className={`${styles.datasetItem} ${selectedDataset?.id === dataset.id ? styles.selected : ''}`}
                onClick={() => setSelectedDataset(dataset)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Database size={16} />
                <div className={styles.datasetInfo}>
                  <span className={styles.datasetName}>{dataset.name}</span>
                  <span className={styles.datasetDetails}>
                    {dataset.samples} samples • {dataset.type}
                  </span>
                </div>
                <span className={`${styles.datasetBadge} ${styles[dataset.type]}`}>
                  {dataset.type}
                </span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Selected Dataset Info */}
        {selectedDataset && (
          <div className={styles.selectedDatasetInfo}>
            <h4>Selected Dataset</h4>
            <div className={styles.datasetStats}>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Name</span>
                <span className={styles.statValue}>{selectedDataset.name}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Target</span>
                <span className={styles.statValue}>{selectedDataset.targetColumn}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Samples</span>
                <span className={styles.statValue}>{selectedDataset.samples}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Type</span>
                <span className={`${styles.statValue} ${styles[selectedDataset.type]}`}>
                  {selectedDataset.type}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className={styles.mainContent}>
        <header className={styles.header}>
          <div className={styles.headerLeft}>
            <h1>Model Comparison Dashboard</h1>
            {selectedDataset && (
              <div className={styles.breadcrumb}>
                <span>Dataset: {selectedDataset.name}</span>
                <span className={styles.separator}>/</span>
                <span>{selectedDataset.type}</span>
              </div>
            )}
          </div>
          
          <div className={styles.headerActions}>
            {trainingResults && (
              <div className={styles.trainingSummary}>
                <div className={styles.summaryBadge}>
                  {Object.values(trainingResults).filter(r => r.status === 'success').length} 
                  models trained
                </div>
              </div>
            )}
            <button className={styles.actionButton}>
              <Download size={16} />
              Export Results
            </button>
            <button className={`${styles.actionButton} ${styles.primary}`}>
              <Home size={16} />
              Home
            </button>
          </div>
        </header>

        <div className={styles.content}>
          {activeTab === 'compare' && (
            <ModelComparison 
              selectedDataset={selectedDataset}
              onTrainingComplete={handleTrainingComplete}
            />
          )}
          
          {activeTab === 'upload' && (
            <DataUpload onDatasetUpload={loadDatasets} />
          )}
          
          {activeTab === 'preprocess' && (
            <PreprocessingPanel dataset={selectedDataset} />
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;