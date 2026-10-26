import React, { useState } from 'react';
import axios from 'axios';
import { Send, RotateCcw } from 'lucide-react';
import RiskGauge from '../components/RiskGauge';
import ShapChart from '../components/ShapChart';
import ExplanationCard from '../components/ExplanationCard';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const DEFAULT_FORM = {
  age: 35,
  income: 65000,
  loan_amount: 15000,
  credit_score: 680,
  employment_years: 5,
  existing_debts: 5000,
  term: '36 months',
  int_rate: 12.5,
  installment: 500,
  dti: 18.5,
  delinq_2yrs: 0,
  revol_bal: 5000,
  revol_util: 45.0,
  pub_rec: 0,
  home_ownership: 'RENT',
  loan_purpose: 'debt_consolidation',
  grade: 'B',
  explanation_mode: 'officer_note',
};

const FIELDS = [
  { key: 'loan_amount',      label: 'Loan Amount ($)',            type: 'number', step: 1000 },
  { key: 'income',           label: 'Annual Income ($)',           type: 'number', step: 1000 },
  { key: 'age',              label: 'Applicant Age',              type: 'number', step: 1    },
  { key: 'credit_score',     label: 'Credit Score',               type: 'number', step: 1    },
  { key: 'employment_years', label: 'Employment Years',           type: 'number', step: 1    },
  { key: 'existing_debts',   label: 'Existing Debts ($)',         type: 'number', step: 100  },
  { key: 'int_rate',         label: 'Interest Rate (%)',          type: 'number', step: 0.1  },
  { key: 'dti',              label: 'Debt-to-Income Ratio',       type: 'number', step: 0.1  },
  { key: 'installment',      label: 'Monthly Installment ($)',    type: 'number', step: 10   },
  { key: 'revol_bal',        label: 'Revolving Balance ($)',      type: 'number', step: 100  },
  { key: 'revol_util',       label: 'Revolving Utilization (%)', type: 'number', step: 1    },
  { key: 'delinq_2yrs',      label: 'Delinquencies (2yr)',        type: 'number', step: 1    },
  { key: 'pub_rec',          label: 'Public Records',             type: 'number', step: 1    },
];

const SELECT_FIELDS = [
  { key: 'term', label: 'Loan Term', options: [
    { value: '36 months', label: '36 months' },
    { value: '60 months', label: '60 months' },
  ]},
  { key: 'grade', label: 'Credit Grade', options: ['A','B','C','D','E','F','G'].map(g => ({ value: g, label: g })) },
  { key: 'home_ownership', label: 'Home Ownership', options: ['RENT','OWN','MORTGAGE','OTHER'].map(v => ({ value: v, label: v })) },
  { key: 'loan_purpose', label: 'Loan Purpose', options: [
    'debt_consolidation','credit_card','home_improvement',
    'major_purchase','medical','small_business','other'
  ].map(p => ({ value: p, label: p.replace(/_/g,' ') }))},
  { key: 'explanation_mode', label: 'Explanation Mode', options: [
    { value: 'officer_note',    label: 'Officer Note'    },
    { value: 'customer_letter', label: 'Customer Letter' },
    { value: 'risk_summary',    label: 'Risk Summary'    },
  ]},
];

function DecisionBanner({ decision, riskLevel, probability }) {
  const colors = { APPROVE: '#10b981', DECLINE: '#ef4444', REVIEW: '#f59e0b' };
  const color = colors[decision] || '#f59e0b';
  return (
    <div style={{
      padding: '16px 20px', borderRadius: 10, marginBottom: 20,
      background: `${color}12`, border: `1px solid ${color}40`,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    }}>
      <div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>Decision</div>
        <div style={{ fontSize: 22, fontWeight: 700, color, marginTop: 2 }}>{decision}</div>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>Risk Level</div>
        <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>{riskLevel}</div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>Default Probability</div>
        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color, marginTop: 2 }}>
          {(probability * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
}

export default function Predict() {
  const [form,    setForm]    = useState(DEFAULT_FORM);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  const handleChange = (key, value) => setForm(prev => ({ ...prev, [key]: value }));

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
        const payload = {
      age:              parseInt(form.age),
      income:           parseFloat(form.income),
      loan_amount:      parseFloat(form.loan_amount),
      credit_score:     parseInt(form.credit_score),
      employment_years: parseInt(form.employment_years),
      existing_debts:   parseFloat(form.existing_debts),
      int_rate:         parseFloat(form.int_rate),
      dti:              parseFloat(form.dti),
      installment:      parseFloat(form.installment),
      revol_bal:        parseFloat(form.revol_bal),
      revol_util:       parseFloat(form.revol_util),
      delinq_2yrs:      parseInt(form.delinq_2yrs),
      pub_rec:          parseInt(form.pub_rec),
      term:             form.term,
      grade:            form.grade,
      home_ownership:   form.home_ownership,
      loan_purpose:     form.loan_purpose,
    };
      const { data } = await axios.post(`${API}/predict`, payload);
      setResult(data);
    } catch (e) {
      const detail = e.response?.data?.detail;
      const msg = Array.isArray(detail)
        ? detail.map(d => d.msg).join(', ')
        : (detail || 'Prediction failed. Check API connection.');
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setForm(DEFAULT_FORM); setResult(null); setError(''); };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Predict</h1>
        <p className="page-subtitle">Submit a loan application for risk scoring and GPT-4 explanation</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '620px 1fr', gap: 24 }}>
        <div className="card">
          <div className="card-title">Loan Application</div>
          <div className="grid-2">
            {FIELDS.map(({ key, label, type, step }) => (
              <div className="form-group" key={key}>
                <label className="form-label">{label}</label>
                <input className="form-input" type={type} step={step}
                  value={form[key]} onChange={e => handleChange(key, e.target.value)} />
              </div>
            ))}
          </div>
          <div className="grid-2">
            {SELECT_FIELDS.map(({ key, label, options }) => (
              <div className="form-group" key={key}>
                <label className="form-label">{label}</label>
                <select className="form-select" value={form[key]}
                  onChange={e => handleChange(key, e.target.value)}>
                  {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            ))}
          </div>
          {error && <div className="alert alert-error" style={{ fontSize: 12 }}>{error}</div>}
          <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
            <button className="btn btn-primary" onClick={handleSubmit} disabled={loading} style={{ flex: 1 }}>
              {loading ? <><div className="spinner" /> Analyzing...</> : <><Send size={15} /> Analyze Risk</>}
            </button>
            <button className="btn btn-ghost" onClick={handleReset}>
              <RotateCcw size={15} /> Reset
            </button>
          </div>
        </div>

        <div>
          {result ? (
            <>
              <DecisionBanner decision={result.decision} riskLevel={result.risk_level} probability={result.probability} />
              <div className="grid-2" style={{ marginBottom: 0 }}>
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                  <div className="card-title" style={{ alignSelf: 'flex-start' }}>Risk Score</div>
                  <RiskGauge probability={result.probability} />
                  {result.prediction_id && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>ID: #{result.prediction_id}</div>
                  )}
                </div>
                <div className="card">
                  <div className="card-title">SHAP Feature Impact</div>
                  <ShapChart shapValues={result.shap_analysis || result.shap_values || {}} />
                </div>
              </div>
              <ExplanationCard explanations={result.explanations || {}} llmData={result.llm_explanation || {}} />
            </>
          ) : (
            <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 300 }}>
              <div className="empty-state">
                <p>Fill in the loan application form and click<br /><strong style={{ color: 'var(--accent)' }}>Analyze Risk</strong> to see results here.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}