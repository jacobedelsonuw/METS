import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Heart, BarChart3, List, Activity } from 'lucide-react';
import './MetabolicSyndromeRiskCalculator.css';

const MetabolicSyndromeRiskCalculator = () => {
  const [patientData, setPatientData] = useState({
    age: 45,
    is_male: true,
    waistCircumference: 94,
    triglycerides: 120,
    hdlCholesterol: 45,
    systolicBP: 125,
    diastolicBP: 82,
    fastingGlucose: 95,
    hba1c: 5.6,
    bmi: 27.5,
    crp: 2.0,
    hasDiabetes: false,
    hasHypertension: false,
    hasHyperlipidemia: false,
  });

  const [riskAssessment, setRiskAssessment] = useState(null);
  const [activeTab, setActiveTab] = useState('inputs');
  const [interventionData, setInterventionData] = useState({
    dailySteps: 0,
    sedentaryMinutes: 0,
    caloriesOut: 0,
    sleepTime: 0,
  });

  const normalRanges = {
    waistCircumference: { male: [0, 102], female: [0, 88], unit: 'cm' },
    triglycerides: [0, 150, 'mg/dL'],
    hdlCholesterol: { male: [40, 100], female: [50, 100], unit: 'mg/dL' },
    systolicBP: [90, 130, 'mmHg'],
    diastolicBP: [60, 85, 'mmHg'],
    fastingGlucose: [70, 100, 'mg/dL'],
    hba1c: [4.0, 5.7, '%'],
    bmi: [18.5, 25, 'kg/m²'],
    crp: [0, 3.0, 'mg/L'],
  };

  const coefficients = {
    steps: {
      fastingGlucose: -0.000836,
      waistCircumference: -0.000022,
      triglycerides: -0.000045,
      hdlCholesterol: 0.000325,
      diastolicBP: -0.000010,
      systolicBP: -0.000044,
      riskAdjustment: -0.001, // Direct risk score reduction per step
    },
    activity_calories: {
      fastingGlucose: -0.001757,
      waistCircumference: 0.004728,
      triglycerides: 0.047533,
      hdlCholesterol: 0.056445,
      diastolicBP: 0.001318,
      systolicBP: 0.000148,
      riskAdjustment: -0.01, // Direct risk score reduction per calorie burned
    },
    sedentary_minutes: {
      fastingGlucose: 0.000342,
      waistCircumference: 0.012089,
      triglycerides: -0.102595,
      hdlCholesterol: -0.132005,
      diastolicBP: 0.000573,
      systolicBP: 0.002436,
      riskAdjustment: 0.02, // Direct risk score increase per sedentary minute
    },
    sleep_time: {
      fastingGlucose: -0.005,
      waistCircumference: 0.012089,
      triglycerides: -0.102595,
      hdlCholesterol: 0.092005,
      diastolicBP: 0.004573,
      systolicBP: 0.002436,
      riskAdjustment: -0.5, // Direct risk score reduction per hour of sleep
    },
  };

  const handleSliderChange = (field, value) => {
    setPatientData(prev => ({ ...prev, [field]: parseFloat(value) }));
  };

  const handleSwitchChange = (field, checked) => {
    setPatientData(prev => ({ ...prev, [field]: checked }));
  };

  const handleInterventionChange = (field, value) => {
    setInterventionData(prev => ({ ...prev, [field]: parseFloat(value) }));
  };

  const isAbnormal = (field, value) => {
    if (field === 'waistCircumference' || field === 'hdlCholesterol') {
      const gender = patientData.is_male ? 'male' : 'female';
      return value < normalRanges[field][gender][0] || value > normalRanges[field][gender][1];
    } else if (normalRanges[field]) {
      return value < normalRanges[field][0] || value > normalRanges[field][1];
    }
    return false;
  };

  const getValueColor = (field, value) => {
    return isAbnormal(field, value) ? 'abnormal' : 'normal';
  };

  const calculatePredictedMeasures = () => {
    const { dailySteps, sedentaryMinutes, caloriesOut, sleepTime } = interventionData;
    return {
      waistCircumference: patientData.waistCircumference +
        (coefficients.steps.waistCircumference * dailySteps +
         coefficients.activity_calories.waistCircumference * caloriesOut +
         coefficients.sedentary_minutes.waistCircumference * sedentaryMinutes +
         coefficients.sleep_time.waistCircumference * sleepTime),
      triglycerides: patientData.triglycerides +
        (coefficients.steps.triglycerides * dailySteps +
         coefficients.activity_calories.triglycerides * caloriesOut +
         coefficients.sedentary_minutes.triglycerides * sedentaryMinutes +
         coefficients.sleep_time.triglycerides * sleepTime),
      hdlCholesterol: patientData.hdlCholesterol +
        (coefficients.steps.hdlCholesterol * dailySteps +
         coefficients.activity_calories.hdlCholesterol * caloriesOut +
         coefficients.sedentary_minutes.hdlCholesterol * sedentaryMinutes +
         coefficients.sleep_time.hdlCholesterol * sleepTime),
      systolicBP: patientData.systolicBP +
        (coefficients.steps.systolicBP * dailySteps +
         coefficients.activity_calories.systolicBP * caloriesOut +
         coefficients.sedentary_minutes.systolicBP * sedentaryMinutes +
         coefficients.sleep_time.systolicBP * sleepTime),
      diastolicBP: patientData.diastolicBP +
        (coefficients.steps.diastolicBP * dailySteps +
         coefficients.activity_calories.diastolicBP * caloriesOut +
         coefficients.sedentary_minutes.diastolicBP * sedentaryMinutes +
         coefficients.sleep_time.diastolicBP * sleepTime),
      fastingGlucose: patientData.fastingGlucose +
        (coefficients.steps.fastingGlucose * dailySteps +
         coefficients.activity_calories.fastingGlucose * caloriesOut +
         coefficients.sedentary_minutes.fastingGlucose * sedentaryMinutes +
         coefficients.sleep_time.fastingGlucose * sleepTime),
    };
  };

  const calculateRisk = () => {
    const data = calculatePredictedMeasures();
    const { dailySteps, sedentaryMinutes, caloriesOut, sleepTime } = interventionData;
    const criteria = [];
    const criteriaImpacts = [];
    let criteriaCount = 0;

    // Metabolic criteria (each contributes 14% to criteriaScore, weighted at 70%)
    const criterionWeight = (1 / 5) * 100 * 0.7; // 14% per criterion
    if ((patientData.is_male && data.waistCircumference >= 102) || (!patientData.is_male && data.waistCircumference >= 88)) {
      criteria.push('Elevated waist circumference');
      criteriaImpacts.push({ name: 'Elevated waist circumference', impact: criterionWeight });
      criteriaCount++;
    }
    if (data.triglycerides >= 150) {
      criteria.push('Elevated triglycerides');
      criteriaImpacts.push({ name: 'Elevated triglycerides', impact: criterionWeight });
      criteriaCount++;
    }
    if ((patientData.is_male && data.hdlCholesterol < 40) || (!patientData.is_male && data.hdlCholesterol < 50)) {
      criteria.push('Reduced HDL cholesterol');
      criteriaImpacts.push({ name: 'Reduced HDL cholesterol', impact: criterionWeight });
      criteriaCount++;
    }
    if (data.systolicBP >= 130 || data.diastolicBP >= 85) {
      criteria.push('Elevated blood pressure');
      criteriaImpacts.push({ name: 'Elevated blood pressure', impact: criterionWeight });
      criteriaCount++;
    }
    if (data.fastingGlucose >= 100) {
      criteria.push('Elevated fasting glucose');
      criteriaImpacts.push({ name: 'Elevated fasting glucose', impact: criterionWeight });
      criteriaCount++;
    }

    const hasMetabolicSyndrome = criteriaCount >= 3;
    const criteriaScore = (criteriaCount / 5) * 100;

    const additionalFactors = [];
    const additionalImpacts = [];
    let additionalRisk = 0;

    // Additional factors (weighted at 30%)
    if (patientData.age > 65) {
      additionalFactors.push('Advanced age');
      additionalImpacts.push({ name: 'Advanced age', impact: 10 * 0.3 });
      additionalRisk += 10;
    } else if (patientData.age > 50) {
      additionalFactors.push('Age over 50');
      additionalImpacts.push({ name: 'Age over 50', impact: 5 * 0.3 });
      additionalRisk += 5;
    }
    if (patientData.bmi >= 30) {
      additionalFactors.push('Obesity (BMI ≥ 30)');
      additionalImpacts.push({ name: 'Obesity (BMI ≥ 30)', impact: 10 * 0.3 });
      additionalRisk += 10;
    } else if (patientData.bmi >= 25) {
      additionalFactors.push('Overweight (BMI ≥ 25)');
      additionalImpacts.push({ name: 'Overweight (BMI ≥ 25)', impact: 5 * 0.3 });
      additionalRisk += 5;
    }
    if (patientData.hba1c >= 6.5) {
      additionalFactors.push('Diabetic range HbA1c');
      additionalImpacts.push({ name: 'Diabetic range HbA1c', impact: 15 * 0.3 });
      additionalRisk += 15;
    } else if (patientData.hba1c >= 5.7) {
      additionalFactors.push('Pre-diabetic range HbA1c');
      additionalImpacts.push({ name: 'Pre-diabetic range HbA1c', impact: 7 * 0.3 });
      additionalRisk += 7;
    }
    if (patientData.hasDiabetes) {
      additionalFactors.push('Diagnosed diabetes');
      additionalImpacts.push({ name: 'Diagnosed diabetes', impact: 15 * 0.3 });
      additionalRisk += 15;
    }
    if (patientData.hasHypertension) {
      additionalFactors.push('Diagnosed hypertension');
      additionalImpacts.push({ name: 'Diagnosed hypertension', impact: 10 * 0.3 });
      additionalRisk += 10;
    }
    if (patientData.hasHyperlipidemia) {
      additionalFactors.push('Diagnosed hyperlipidemia');
      additionalImpacts.push({ name: 'Diagnosed hyperlipidemia', impact: 8 * 0.3 });
      additionalRisk += 8;
    }
    if (patientData.crp > 3) {
      additionalFactors.push('Elevated C-reactive protein');
      additionalImpacts.push({ name: 'Elevated C-reactive protein', impact: 5 * 0.3 });
      additionalRisk += 5;
    }

    // Base risk score from criteria and additional factors
    let riskScore = 0.7 * criteriaScore + 0.3 * additionalRisk;

    // Integrate interventions into the risk score
    const interventionAdjustment =
      (coefficients.steps.riskAdjustment * dailySteps) +
      (coefficients.activity_calories.riskAdjustment * caloriesOut) +
      (coefficients.sedentary_minutes.riskAdjustment * sedentaryMinutes) +
      (coefficients.sleep_time.riskAdjustment * sleepTime);

    const interventionImpacts = [];
    if (dailySteps > 0) {
      interventionImpacts.push({
        name: `Daily Steps (+${dailySteps})`,
        impact: coefficients.steps.riskAdjustment * dailySteps,
      });
    }
    if (sedentaryMinutes > 0) {
      interventionImpacts.push({
        name: `Sedentary Minutes (-${sedentaryMinutes})`,
        impact: coefficients.sedentary_minutes.riskAdjustment * sedentaryMinutes,
      });
    }
    if (caloriesOut > 0) {
      interventionImpacts.push({
        name: `Calories Burned (+${caloriesOut})`,
        impact: coefficients.activity_calories.riskAdjustment * caloriesOut,
      });
    }
    if (sleepTime > 0) {
      interventionImpacts.push({
        name: `Sleep Time (+${sleepTime} hrs)`,
        impact: coefficients.sleep_time.riskAdjustment * sleepTime,
      });
    }

    riskScore = Math.max(0, Math.min(100, riskScore + interventionAdjustment));

    const riskCategory = riskScore < 30 ? 'Low' : riskScore < 60 ? 'Moderate' : 'High';
    const recommendation = riskCategory === 'High' || hasMetabolicSyndrome
      ? 'Comprehensive metabolic evaluation and aggressive lifestyle intervention are recommended.'
      : riskCategory === 'Moderate'
        ? 'Lifestyle modifications focusing on diet and exercise are recommended.'
        : 'Maintain a healthy lifestyle.';

    const interventionComments = interventionImpacts.map(
      ({ name, impact }) => `${name} adjusts risk by ${impact.toFixed(2)}%.`
    ) || ['No interventions applied.'];

    setRiskAssessment({
      riskScore: Math.round(riskScore),
      riskCategory,
      hasMetabolicSyndrome,
      criteriaCount,
      criteria,
      criteriaImpacts,
      additionalFactors,
      additionalImpacts,
      recommendation,
      interventionComments,
      interventionImpacts,
      predictedMeasures: data,
    });
    setActiveTab('results');
  };

  const getChartData = () => {
    if (!riskAssessment) return [];
    const dataToUse = riskAssessment.predictedMeasures || patientData;
    const normalizedData = [
      { name: 'Waist', value: dataToUse.waistCircumference, threshold: patientData.is_male ? 102 : 88, unit: 'cm' },
      { name: 'Triglycerides', value: dataToUse.triglycerides, threshold: 150, unit: 'mg/dL' },
      { name: 'HDL', value: dataToUse.hdlCholesterol, threshold: patientData.is_male ? 40 : 50, unit: 'mg/dL', inverted: true },
      { name: 'Systolic BP', value: dataToUse.systolicBP, threshold: 130, unit: 'mmHg' },
      { name: 'Diastolic BP', value: dataToUse.diastolicBP, threshold: 85, unit: 'mmHg' },
      { name: 'Glucose', value: dataToUse.fastingGlucose, threshold: 100, unit: 'mg/dL' },
    ];
    return normalizedData.map(item => ({
      ...item,
      percentOfThreshold: Math.min(150, item.inverted ? (item.threshold / item.value) * 100 : (item.value / item.threshold) * 100),
      metCriterion: item.inverted ? item.value < item.threshold : item.value >= item.threshold,
    }));
  };

  const resetForm = () => {
    setPatientData({
      age: 45,
      is_male: true,
      waistCircumference: 94,
      triglycerides: 120,
      hdlCholesterol: 45,
      systolicBP: 125,
      diastolicBP: 82,
      fastingGlucose: 95,
      hba1c: 5.6,
      bmi: 27.5,
      crp: 2.0,
      hasDiabetes: false,
      hasHypertension: false,
      hasHyperlipidemia: false,
    });
    setInterventionData({ dailySteps: 0, sedentaryMinutes: 0, caloriesOut: 0, sleepTime: 0 });
    setRiskAssessment(null);
    setActiveTab('inputs');
  };

  const loadPatientProfile = (profile) => {
    const profiles = {
      'low-risk': {
        age: 32, is_male: false, waistCircumference: 80, triglycerides: 95,
        hdlCholesterol: 62, systolicBP: 118, diastolicBP: 75, fastingGlucose: 88,
        hba1c: 5.2, bmi: 23.5, crp: 1.2, hasDiabetes: false, hasHypertension: false, hasHyperlipidemia: false,
      },
      'moderate-risk': {
        age: 58, is_male: true, waistCircumference: 100, triglycerides: 162,
        hdlCholesterol: 38, systolicBP: 132, diastolicBP: 82, fastingGlucose: 106,
        hba1c: 5.8, bmi: 28.3, crp: 2.8, hasDiabetes: false, hasHypertension: true, hasHyperlipidemia: false,
      },
      'high-risk': {
        age: 62, is_male: false, waistCircumference: 110, triglycerides: 218,
        hdlCholesterol: 42, systolicBP: 148, diastolicBP: 92, fastingGlucose: 125,
        hba1c: 6.4, bmi: 33.1, crp: 4.5, hasDiabetes: true, hasHypertension: true, hasHyperlipidemia: true,
      },
    };
    setPatientData(profiles[profile] || patientData);
  };

  return (
    <div className="calculator-container">
      <div className="header">
        <h1 className="title">
          <Heart className="icon" /> Metabolic Syndrome Risk Calculator
        </h1>
        <p className="description">Assess your risk of metabolic syndrome based on clinical measurements</p>
      </div>

      <div className="tabs">
        <div className="tab-list">
          <button className={`tab-button ${activeTab === 'inputs' ? 'active' : ''}`} onClick={() => setActiveTab('inputs')}>
            <List className="icon" /> Patient Data
          </button>
          <button className={`tab-button ${activeTab === 'results' && riskAssessment ? 'active' : ''}`} onClick={() => setActiveTab('results')} disabled={!riskAssessment}>
            <BarChart3 className="icon" /> Risk Assessment
          </button>
          <button className={`tab-button ${activeTab === 'chart' && riskAssessment ? 'active' : ''}`} onClick={() => setActiveTab('chart')} disabled={!riskAssessment}>
            <Activity className="icon" /> Visualization
          </button>
          <button className={`tab-button ${activeTab === 'intervention' && riskAssessment ? 'active' : ''}`} onClick={() => setActiveTab('intervention')} disabled={!riskAssessment}>
            <Activity className="icon" /> Intervention
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'inputs' && (
            <div className="inputs-section">
              <div className="grid">
                <div>
                  <h3>Demographics</h3>
                  <div className="input-group">
                    <label>Age: {patientData.age} years</label>
                    <input
                      type="range"
                      min="18"
                      max="90"
                      step="1"
                      value={patientData.age}
                      onChange={(e) => handleSliderChange('age', e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Male</label>
                    <input
                      type="checkbox"
                      checked={patientData.is_male}
                      onChange={(e) => handleSwitchChange('is_male', e.target.checked)}
                    />
                  </div>
                </div>
                <div>
                  <h3>Core Measurements</h3>
                  <div className="input-group">
                    <label className={getValueColor('waistCircumference', patientData.waistCircumference)}>
                      Waist Circumference: {patientData.waistCircumference} cm
                    </label>
                    <input
                      type="range"
                      min="60"
                      max="150"
                      step="1"
                      value={patientData.waistCircumference}
                      onChange={(e) => handleSliderChange('waistCircumference', e.target.value)}
                    />
                    <small>Normal: ≤{patientData.is_male ? '102' : '88'} cm</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('triglycerides', patientData.triglycerides)}>
                      Triglycerides: {patientData.triglycerides} mg/dL
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="300"
                      step="1"
                      value={patientData.triglycerides}
                      onChange={(e) => handleSliderChange('triglycerides', e.target.value)}
                    />
                    <small>Normal: ≤150 mg/dL</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('hdlCholesterol', patientData.hdlCholesterol)}>
                      HDL Cholesterol: {patientData.hdlCholesterol} mg/dL
                    </label>
                    <input
                      type="range"
                      min="20"
                      max="100"
                      step="1"
                      value={patientData.hdlCholesterol}
                      onChange={(e) => handleSliderChange('hdlCholesterol', e.target.value)}
                    />
                    <small>Normal: ≥{patientData.is_male ? '40' : '50'} mg/dL</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('systolicBP', patientData.systolicBP)}>
                      Systolic BP: {patientData.systolicBP} mmHg
                    </label>
                    <input
                      type="range"
                      min="90"
                      max="180"
                      step="1"
                      value={patientData.systolicBP}
                      onChange={(e) => handleSliderChange('systolicBP', e.target.value)}
                    />
                    <small>Normal: ≤130 mmHg</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('diastolicBP', patientData.diastolicBP)}>
                      Diastolic BP: {patientData.diastolicBP} mmHg
                    </label>
                    <input
                      type="range"
                      min="60"
                      max="120"
                      step="1"
                      value={patientData.diastolicBP}
                      onChange={(e) => handleSliderChange('diastolicBP', e.target.value)}
                    />
                    <small>Normal: ≤85 mmHg</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('fastingGlucose', patientData.fastingGlucose)}>
                      Fasting Glucose: {patientData.fastingGlucose} mg/dL
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="200"
                      step="1"
                      value={patientData.fastingGlucose}
                      onChange={(e) => handleSliderChange('fastingGlucose', e.target.value)}
                    />
                    <small>Normal: ≤100 mg/dL</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('hba1c', patientData.hba1c)}>
                      HbA1c: {patientData.hba1c} %
                    </label>
                    <input
                      type="range"
                      min="4"
                      max="10"
                      step="0.1"
                      value={patientData.hba1c}
                      onChange={(e) => handleSliderChange('hba1c', e.target.value)}
                    />
                    <small>Normal: ≤5.7 %</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('bmi', patientData.bmi)}>
                      BMI: {patientData.bmi} kg/m²
                    </label>
                    <input
                      type="range"
                      min="15"
                      max="40"
                      step="0.1"
                      value={patientData.bmi}
                      onChange={(e) => handleSliderChange('bmi', e.target.value)}
                    />
                    <small>Normal: 18.5-25 kg/m²</small>
                  </div>
                  <div className="input-group">
                    <label className={getValueColor('crp', patientData.crp)}>
                      CRP: {patientData.crp} mg/L
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="10"
                      step="0.1"
                      value={patientData.crp}
                      onChange={(e) => handleSliderChange('crp', e.target.value)}
                    />
                    <small>Normal: ≤3 mg/L</small>
                  </div>
                  <div className="input-group">
                    <label>Diabetes</label>
                    <input
                      type="checkbox"
                      checked={patientData.hasDiabetes}
                      onChange={(e) => handleSwitchChange('hasDiabetes', e.target.checked)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Hypertension</label>
                    <input
                      type="checkbox"
                      checked={patientData.hasHypertension}
                      onChange={(e) => handleSwitchChange('hasHypertension', e.target.checked)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Hyperlipidemia</label>
                    <input
                      type="checkbox"
                      checked={patientData.hasHyperlipidemia}
                      onChange={(e) => handleSwitchChange('hasHyperlipidemia', e.target.checked)}
                    />
                  </div>
                </div>
              </div>
              <div className="profile-buttons">
                <button onClick={() => loadPatientProfile('low-risk')}>Load Low Risk Profile</button>
                <button onClick={() => loadPatientProfile('moderate-risk')}>Load Moderate Risk Profile</button>
                <button onClick={() => loadPatientProfile('high-risk')}>Load High Risk Profile</button>
              </div>
            </div>
          )}

          {activeTab === 'results' && riskAssessment && (
            <div className="results-section">
              <div className="risk-score">
                <h3>Risk Score: {riskAssessment.riskScore}%</h3>
                <p className={riskAssessment.riskCategory.toLowerCase()}>{riskAssessment.riskCategory} Risk</p>
              </div>
              <div className="metabolic-status">
                <h3>Metabolic Syndrome ({riskAssessment.criteriaCount}/5 criteria)</h3>
                <p>{riskAssessment.hasMetabolicSyndrome ? 'Present' : 'Absent'}</p>
                <ul>
                  {riskAssessment.criteriaImpacts.map((item, index) => (
                    <li key={index}>{item.name} (+{item.impact.toFixed(2)}%)</li>
                  ))}
                </ul>
              </div>
              <div className="additional-factors">
                <h3>Additional Risk Factors</h3>
                <ul>
                  {riskAssessment.additionalImpacts.map((item, index) => (
                    <li key={index}>{item.name} (+{item.impact.toFixed(2)}%)</li>
                  ))}
                </ul>
              </div>
              <div className="intervention-comments">
                <h3>Intervention Impact</h3>
                <ul>
                  {riskAssessment.interventionImpacts.map((item, index) => (
                    <li key={index}>{item.name} ({item.impact >= 0 ? '+' : ''}{item.impact.toFixed(2)}%)</li>
                  ))}
                </ul>
              </div>
              <div className="recommendation">
                <h3>Recommendation</h3>
                <p>{riskAssessment.recommendation}</p>
              </div>
            </div>
          )}

          {activeTab === 'chart' && riskAssessment && (
            <div className="chart-section">
              <h3>Risk Factor Visualization</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={getChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="percentOfThreshold" stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === 'intervention' && riskAssessment && (
            <div className="intervention-section">
              <h3>Lifestyle Interventions</h3>
              <div className="input-group">
                <label>Increase Daily Steps: {interventionData.dailySteps}</label>
                <input
                  type="range"
                  min="0"
                  max="10000"
                  step="500"
                  value={interventionData.dailySteps}
                  onChange={(e) => handleInterventionChange('dailySteps', e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>Reduce Sedentary Minutes: {interventionData.sedentaryMinutes}</label>
                <input
                  type="range"
                  min="0"
                  max="600"
                  step="30"
                  value={interventionData.sedentaryMinutes}
                  onChange={(e) => handleInterventionChange('sedentaryMinutes', e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>Increase Calories Burned: {interventionData.caloriesOut}</label>
                <input
                  type="range"
                  min="0"
                  max="1000"
                  step="50"
                  value={interventionData.caloriesOut}
                  onChange={(e) => handleInterventionChange('caloriesOut', e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>Increase Sleep Time (hours): {interventionData.sleepTime}</label>
                <input
                  type="range"
                  min="0"
                  max="12"
                  step="0.5"
                  value={interventionData.sleepTime}
                  onChange={(e) => handleInterventionChange('sleepTime', e.target.value)}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="footer">
        <button onClick={resetForm}>Reset</button>
        <button onClick={calculateRisk}>Calculate Risk</button>
      </div>

      <p className="disclaimer">This calculator is for educational purposes only and should not replace professional medical advice.</p>
    </div>
  );
};

export default MetabolicSyndromeRiskCalculator;
