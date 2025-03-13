import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Heart, BarChart3, List, Activity } from 'lucide-react';

const MetabolicSyndromeRiskCalculator = () => {
  // Initial patient data state
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
    hasHyperlipidemia: false
  });

  const [riskAssessment, setRiskAssessment] = useState(null);
  const [activeTab, setActiveTab] = useState('inputs');
  const [interventionData, setInterventionData] = useState({
    dailySteps: 0,
    sedentaryMinutes: 0,
    caloriesOut: 0,
    sleepTime: 0
  });

  // Normal ranges reference
  const normalRanges = {
    waistCircumference: { male: [0, 102], female: [0, 88], unit: 'cm' },
    triglycerides: [0, 150, 'mg/dL'],
    hdlCholesterol: { male: [40, 100], female: [50, 100], unit: 'mg/dL' },
    systolicBP: [90, 130, 'mmHg'],
    diastolicBP: [60, 85, 'mmHg'],
    fastingGlucose: [70, 100, 'mg/dL'],
    hba1c: [4.0, 5.7, '%'],
    bmi: [18.5, 25, 'kg/m²'],
    crp: [0, 3.0, 'mg/L']
  };

  // Correlation coefficients
  const coefficients = {
    steps: {
      fastingGlucose: -0.000836,
      waistCircumference: -0.000022,
      triglycerides: -0.000045,
      hdlCholesterol: 0.000325,
      diastolicBP: -0.000010,
      systolicBP: -0.000044
    },
    activity_calories: {
      fastingGlucose: -0.001757,
      waistCircumference: 0.004728,
      triglycerides: 0.047533,
      hdlCholesterol: 0.056445,
      diastolicBP: 0.001318,
      systolicBP: 0.000148
    },
    sedentary_minutes: {
      fastingGlucose: 0.000342,
      waistCircumference: 0.012089,
      triglycerides: -0.102595,
      hdlCholesterol: -0.132005,
      diastolicBP: 0.000573,
      systolicBP: 0.002436
    },
    sleep_time: {
      fastingGlucose: -0.005,
      waistCircumference: 0.012089,
      triglycerides: -0.102595,
      hdlCholesterol: 0.092005,
      diastolicBP: 0.004573,
      systolicBP: 0.002436
    }
  };

  // Handler functions
  const handleSliderChange = (field, value) => {
    setPatientData(prev => ({ ...prev, [field]: value[0] || value }));
  };

  const handleSwitchChange = (field, checked) => {
    setPatientData(prev => ({ ...prev, [field]: checked }));
  };

  const handleInterventionChange = (field, value) => {
    setInterventionData(prev => ({ ...prev, [field]: value[0] || value }));
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
    return isAbnormal(field, value) ? 'text-red-500 font-bold' : 'text-green-500';
  };

  const calculatePredictedMeasures = () => {
    const { dailySteps, sedentaryMinutes, caloriesOut, sleepTime } = interventionData;
    
    const predictedMeasures = {
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
         coefficients.sleep_time.fastingGlucose * sleepTime)
    };
    return predictedMeasures;
  };

  const calculateRisk = (usePredicted = false) => {
    const data = usePredicted ? calculatePredictedMeasures() : patientData;
    const criteria = [];
    let criteriaCount = 0;

    // Metabolic syndrome criteria checks
    if ((patientData.is_male && data.waistCircumference >= 102) || 
        (!patientData.is_male && data.waistCircumference >= 88)) {
      criteria.push('Elevated waist circumference');
      criteriaCount++;
    }

    if (data.triglycerides >= 150) {
      criteria.push('Elevated triglycerides');
      criteriaCount++;
    }

    if ((patientData.is_male && data.hdlCholesterol < 40) || 
        (!patientData.is_male && data.hdlCholesterol < 50)) {
      criteria.push('Reduced HDL cholesterol');
      criteriaCount++;
    }

    if (data.systolicBP >= 130 || data.diastolicBP >= 85) {
      criteria.push('Elevated blood pressure');
      criteriaCount++;
    }

    if (data.fastingGlucose >= 100) {
      criteria.push('Elevated fasting glucose');
      criteriaCount++;
    }

    const hasMetabolicSyndrome = criteriaCount >= 3;
    const criteriaScore = (criteriaCount / 5) * 100;

    // Additional risk factors
    const additionalFactors = [];
    let additionalRisk = 0;

    if (patientData.age > 65) {
      additionalFactors.push('Advanced age');
      additionalRisk += 10;
    } else if (patientData.age > 50) {
      additionalFactors.push('Age over 50');
      additionalRisk += 5;
    }

    if (patientData.bmi >= 30) {
      additionalFactors.push('Obesity (BMI ≥ 30)');
      additionalRisk += 10;
    } else if (patientData.bmi >= 25) {
      additionalFactors.push('Overweight (BMI ≥ 25)');
      additionalRisk += 5;
    }

    if (patientData.hba1c >= 6.5) {
      additionalFactors.push('Diabetic range HbA1c');
      additionalRisk += 15;
    } else if (patientData.hba1c >= 5.7) {
      additionalFactors.push('Pre-diabetic range HbA1c');
      additionalRisk += 7;
    }

    if (patientData.hasDiabetes) {
      additionalFactors.push('Diagnosed diabetes');
      additionalRisk += 15;
    }
    if (patientData.hasHypertension) {
      additionalFactors.push('Diagnosed hypertension');
      additionalRisk += 10;
    }
    if (patientData.hasHyperlipidemia) {
      additionalFactors.push('Diagnosed hyperlipidemia');
      additionalRisk += 8;
    }
    if (patientData.crp > 3) {
      additionalFactors.push('Elevated C-reactive protein');
      additionalRisk += 5;
    }

    const riskScore = Math.min(100, 0.7 * criteriaScore + 0.3 * additionalRisk);
    const { riskCategory, riskColor } = riskScore < 30 
      ? { riskCategory: 'Low', riskColor: 'bg-green-500' }
      : riskScore < 60 
        ? { riskCategory: 'Moderate', riskColor: 'bg-yellow-500' }
        : { riskCategory: 'High', riskColor: 'bg-red-500' };

    const recommendation = riskCategory === 'High' || hasMetabolicSyndrome
      ? 'Comprehensive metabolic evaluation and aggressive lifestyle intervention are recommended. Consider medication therapy for specific abnormal parameters. Follow-up within 1-3 months.'
      : riskCategory === 'Moderate'
        ? 'Lifestyle modifications focusing on diet and exercise are recommended. Monitor abnormal parameters closely. Follow-up within 3-6 months.'
        : 'Maintain a healthy lifestyle. Continue routine annual screening.';

    setRiskAssessment({
      riskScore: Math.round(riskScore),
      riskCategory,
      riskColor,
      hasMetabolicSyndrome,
      criteriaCount,
      criteria,
      additionalFactors,
      recommendation,
      parameters: { ...patientData, criteriaCount }
    });
    setActiveTab('results');
  };

  const recalculateRiskScore = () => calculateRisk(true);

  const getChartData = () => {
    if (!riskAssessment) return [];
    
    const normalizedData = [
      { name: 'Waist', value: patientData.waistCircumference, threshold: patientData.is_male ? 102 : 88, unit: 'cm' },
      { name: 'Triglycerides', value: patientData.triglycerides, threshold: 150, unit: 'mg/dL' },
      { name: 'HDL', value: patientData.hdlCholesterol, threshold: patientData.is_male ? 40 : 50, unit: 'mg/dL', inverted: true },
      { name: 'Systolic BP', value: patientData.systolicBP, threshold: 130, unit: 'mmHg' },
      { name: 'Diastolic BP', value: patientData.diastolicBP, threshold: 85, unit: 'mmHg' },
      { name: 'Glucose', value: patientData.fastingGlucose, threshold: 100, unit: 'mg/dL' }
    ];

    return normalizedData.map(item => ({
      ...item,
      percentOfThreshold: Math.min(150, item.inverted 
        ? (item.threshold / item.value) * 100
        : (item.value / item.threshold) * 100),
      metCriterion: item.inverted ? item.value < item.threshold : item.value >= item.threshold
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
      hasHyperlipidemia: false
    });
    setInterventionData({
      dailySteps: 0,
      sedentaryMinutes: 0,
      caloriesOut: 0,
      sleepTime: 0
    });
    setRiskAssessment(null);
    setActiveTab('inputs');
  };

  const loadPatientProfile = (profile) => {
    const profiles = {
      'low-risk': {
        age: 32, is_male: false, waistCircumference: 80, triglycerides: 95,
        hdlCholesterol: 62, systolicBP: 118, diastolicBP: 75, fastingGlucose: 88,
        hba1c: 5.2, bmi: 23.5, crp: 1.2, hasDiabetes: false,
        hasHypertension: false, hasHyperlipidemia: false
      },
      'moderate-risk': {
        age: 58, is_male: true, waistCircumference: 100, triglycerides: 162,
        hdlCholesterol: 38, systolicBP: 132, diastolicBP: 82, fastingGlucose: 106,
        hba1c: 5.8, bmi: 28.3, crp: 2.8, hasDiabetes: false,
        hasHypertension: true, hasHyperlipidemia: false
      },
      'high-risk': {
        age: 62, is_male: false, waistCircumference: 110, triglycerides: 218,
        hdlCholesterol: 42, systolicBP: 148, diastolicBP: 92, fastingGlucose: 125,
        hba1c: 6.4, bmi: 33.1, crp: 4.5, hasDiabetes: true,
        hasHypertension: true, hasHyperlipidemia: true
      }
    };
    setPatientData(profiles[profile] || patientData);
  };

  // Render function remains mostly the same as original
  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <Card className="shadow-lg">
        <CardHeader className="bg-gradient-to-r from-blue-600 to-blue-800 text-white">
          <CardTitle className="text-2xl flex items-center">
            <Heart className="mr-2" /> Metabolic Syndrome Risk Calculator
          </CardTitle>
          <CardDescription className="text-blue-100">
            Assess your risk of metabolic syndrome based on clinical measurements
          </CardDescription>
        </CardHeader>
        
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="px-6 pt-4">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="inputs" className="flex items-center">
                <List className="mr-2 h-4 w-4" />
                <span>Patient Data</span>
              </TabsTrigger>
              <TabsTrigger value="results" className="flex items-center" disabled={!riskAssessment}>
                <BarChart3 className="mr-2 h-4 w-4" />
                <span>Risk Assessment</span>
              </TabsTrigger>
              <TabsTrigger value="chart" className="flex items-center" disabled={!riskAssessment}>
                <Activity className="mr-2 h-4 w-4" />
                <span>Visualization</span>
              </TabsTrigger>
              <TabsTrigger value="intervention" className="flex items-center" disabled={!riskAssessment}>
                <Activity className="mr-2 h-4 w-4" />
                <span>Intervention</span>
              </TabsTrigger>
            </TabsList>
          </div>

          <CardContent className="pt-4">
            <TabsContent value="inputs" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium mb-4">Demographics</h3>
                    <div className="grid gap-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label htmlFor="age">Age</Label>
                          <span>{patientData.age} years</span>
                        </div>
                        <Slider
                          id="age"
                          min={18}
                          max={90}
                          step={1}
                          value={[patientData.age]}
                          onValueChange={(value) => handleSliderChange('age', value)}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="gender">Male</Label>
                        <Switch
                          id="gender"
                          checked={patientData.is_male}
                          onCheckedChange={(checked) => handleSwitchChange('is_male', checked)}
                        />
                      </div>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-medium mb-4">Core Measurements</h3>
                    <div className="grid gap-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label htmlFor="waist">Waist Circumference</Label>
                          <span className={getValueColor('waistCircumference', patientData.waistCircumference)}>
                            {patientData.waistCircumference} cm
                          </span>
                        </div>
                        <Slider
                          id="waist"
                          min={60}
                          max={150}
                          step={1}
                          value={[patientData.waistCircumference]}
                          onValueChange={(value) => handleSliderChange('waistCircumference', value)}
                        />
                        <div className="text-xs text-gray-500">
                          Normal: ≤{patientData.is_male ? '102' : '88'} cm
                        </div>
                      </div>
                      {/* Add other sliders similarly */}
                    </div>
                  </div>
                </div>
                {/* Right column inputs */}
              </div>
            </TabsContent>
            {/* Other TabsContent sections */}
          </CardContent>
        </Tabs>
        
        <CardFooter className="flex flex-col sm:flex-row justify-between gap-4 pt-0">
          <Button variant="outline" onClick={resetForm}>
            Reset
          </Button>
          <Button onClick={() => calculateRisk(false)} className="w-full sm:w-auto">
            Calculate Risk
          </Button>
        </CardFooter>
      </Card>
      
      <div className="mt-4 text-center text-xs text-gray-500">
        This calculator is for educational purposes only and should not replace professional medical advice.
      </div>
    </div>
  );
};

export default MetabolicSyndromeRiskCalculator;
