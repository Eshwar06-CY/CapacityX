import React, { useState, useEffect } from 'react';

// Default scenarios for immediate testability across the 5 tools
const PRESET_SCENARIOS = [
  {
    id: 'scenario-1',
    name: 'Cold-Chain Breakdown (Replacement Ready)',
    desc: 'Refrigerated transport engine failure on highway; depot backup truck on standby.',
    data: {
      shipmentId: 'SHP-8921-COLD',
      deliveryDeadline: '2026-10-20T14:30:00Z',
      truckBreakdown: true,
      breakdownSeverity: 'CRITICAL',
      replacementVehicleAvailable: true,
      customerNotified: false,
      cargoType: 'Cold-Chain Perishables (Frozen Seafood)',
      origin: 'Milwaukee Cold Depot',
      destination: 'Chicago Central Distribution Hub',
      estimatedDistanceKm: 42.0,
    },
  },
  {
    id: 'scenario-2',
    name: 'Dry Freight Breakdown (No Replacement)',
    desc: 'Transmission fault in rural zone; no available backup units nearby.',
    data: {
      shipmentId: 'SHP-4012-DRY',
      deliveryDeadline: '2026-10-20T18:00:00Z',
      truckBreakdown: true,
      breakdownSeverity: 'MODERATE',
      replacementVehicleAvailable: false,
      customerNotified: false,
      cargoType: 'Standard Freight (Consumer Electronics)',
      origin: 'Indianapolis Hub',
      destination: 'Peoria Logistics Center',
      estimatedDistanceKm: 115.0,
    },
  },
  {
    id: 'scenario-3',
    name: 'Urgent Medical (Normal Transit, Notified)',
    desc: 'Time-sensitive pharmaceuticals on schedule; telemetry nominal.',
    data: {
      shipmentId: 'SHP-1109-MED',
      deliveryDeadline: '2026-10-20T12:00:00Z',
      truckBreakdown: false,
      breakdownSeverity: 'NONE',
      replacementVehicleAvailable: true,
      customerNotified: true,
      cargoType: 'Pharmaceuticals & Vaccines',
      origin: 'Madison Biotech Depot',
      destination: 'Rockford Memorial Hospital',
      estimatedDistanceKm: 30.5,
    },
  },
];

const BACKEND_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8081';

const getBackendPortOrHost = (urlStr) => {
  try {
    const parsed = new URL(urlStr);
    return parsed.port || parsed.host;
  } catch {
    return '8081';
  }
};

const backendPortDisplay = getBackendPortOrHost(BACKEND_BASE_URL);

export default function App() {
  const [formData, setFormData] = useState(PRESET_SCENARIOS[0].data);
  const [activeScenarioId, setActiveScenarioId] = useState('scenario-1');
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', message: 'Checking backend...' });
  const [modelStatus, setModelStatus] = useState({ status: 'checking', is_real_model_available: false });
  const [isLoading, setIsLoading] = useState(false);
  const [decisionResult, setDecisionResult] = useState(null);
  const [showJsonRaw, setShowJsonRaw] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setBackendHealth({
          status: 'online',
          message: `Connected (${BACKEND_BASE_URL})`,
          engine: data.ai_engine,
        });
        if (data.strands_decider) {
          setModelStatus(data.strands_decider);
        }
      } else {
        setBackendHealth({ status: 'offline', message: `Backend responded with error at ${BACKEND_BASE_URL}` });
        setModelStatus({ status: 'offline', is_real_model_available: false });
      }
    } catch {
      setBackendHealth({
        status: 'offline',
        message: `Backend offline at ${BACKEND_BASE_URL} (run uvicorn in PowerShell)`,
      });
      setModelStatus({ status: 'offline', is_real_model_available: false });
    }
  };

  const handleScenarioSelect = (scenario) => {
    setActiveScenarioId(scenario.id);
    setFormData({ ...scenario.data });
    setDecisionResult(null);
  };

  const handleFieldChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setActiveScenarioId('custom');
  };

  const handleSimulateDecision = async (e) => {
    e?.preventDefault();
    setIsLoading(true);

    const payload = {
      shipment_id: formData.shipmentId,
      delivery_deadline: formData.deliveryDeadline,
      truck_breakdown: formData.truckBreakdown,
      breakdown_severity: formData.breakdownSeverity,
      replacement_vehicle_available: formData.replacementVehicleAvailable,
      customer_notified: formData.customerNotified,
      cargo_type: formData.cargoType,
      origin: formData.origin,
      destination: formData.destination,
      estimated_distance_remaining_km: parseFloat(formData.estimatedDistanceKm) || 0,
    };

    try {
      const res = await fetch(`${BACKEND_BASE_URL}/api/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        setDecisionResult(data);
      } else {
        generateLocalFallback(payload);
      }
    } catch {
      generateLocalFallback(payload);
    } finally {
      setIsLoading(false);
    }
  };

  // Local fallback when FastAPI backend server is not running
  const generateLocalFallback = (payload) => {
    const isPerishable = /cold|perish|pharma|food|vaccine/i.test(payload.cargo_type || '');
    let selected_action = 'continue_monitoring';
    let urgency = 'LOW';
    let headline = '[Safe Fallback] Continue telemetry monitoring: Transit nominal.';
    let estimated_delay = 0;
    let deadline_met = true;
    let sla_risk = 'LOW';
    let customer_rec = payload.customer_notified ? 'Customer alerted.' : 'Standard updates.';
    let rationale = [
      'SAFE FALLBACK APPLIED: Backend offline or unreachable.',
      'No vehicle breakdown detected.',
      `Transit proceeding toward ${payload.destination}.`,
    ];
    let actions_preview = [
      {
        system: 'Fleet Geofencing',
        action: 'Maintain automated telemetry heartbeat',
        status: 'PREVIEW (Safe Fallback)',
        is_live_action: false,
      },
    ];

    if (payload.truck_breakdown && payload.replacement_vehicle_available) {
      selected_action = 'dispatch_replacement';
      urgency = isPerishable ? 'CRITICAL' : 'HIGH';
      headline = `[Safe Fallback] Dispatch replacement vehicle to intercept shipment ${payload.shipment_id}`;
      estimated_delay = isPerishable ? 35 : 50;
      deadline_met = true;
      sla_risk = 'MODERATE';
      customer_rec = payload.customer_notified ? 'Customer already notified.' : 'URGENT: Alert customer.';
      rationale = [
        'SAFE FALLBACK APPLIED: Backend offline or unreachable.',
        `Breakdown reported (${payload.breakdown_severity}); replacement asset available.`,
        'Mandatory human review required in fallback mode.',
      ];
      actions_preview = [
        {
          system: 'Fleet Dispatch',
          action: `Lock backup unit for shipment ${payload.shipment_id}`,
          status: 'PREVIEW (Safe Fallback)',
          is_live_action: false,
        },
      ];
    } else if (payload.truck_breakdown && !payload.replacement_vehicle_available) {
      selected_action = 'escalate_to_human';
      urgency = 'CRITICAL';
      headline = `[Safe Fallback] Escalate to human: Vehicle disabled with NO local backup vehicles.`;
      estimated_delay = 180;
      deadline_met = false;
      sla_risk = 'CRITICAL_BREACH';
      customer_rec = 'IMMEDIATE ACTION: Dispatch customer success SLA breach alert.';
      rationale = [
        'SAFE FALLBACK APPLIED: Backend offline or unreachable.',
        'Vehicle breakdown with zero backup assets available in range.',
        'Human dispatcher intervention mandatory.',
      ];
      actions_preview = [
        {
          system: 'Emergency Logistics',
          action: 'Contact third-party heavy tow service',
          status: 'PREVIEW (Safe Fallback)',
          is_live_action: false,
        },
      ];
    }

    setDecisionResult({
      is_simulated: true,
      real_action_taken: false,
      environment: 'LOCAL_SAFE_FALLBACK',
      disclaimer: 'SAFE FALLBACK APPLIED: Backend was unreachable. Output generated by deterministic safety rules.',
      shipment_id: payload.shipment_id,
      timestamp: new Date().toISOString(),
      outcome: {
        selected_action,
        action_type: selected_action.toUpperCase(),
        model_confidence: null,
        alternatives: {},
        human_review_required: true,
        engine: 'SAFE_FALLBACK',
        is_real_model: false,
        fallback_reason: 'Backend offline or unreachable. Client-side safe policy engaged.',
        model_id: 'none',
        urgency,
        headline,
        rationale,
        actions_preview,
        estimated_delay_minutes: estimated_delay,
        deadline_met_projected: deadline_met,
        sla_risk_level: sla_risk,
        customer_status_recommended: customer_rec,
      },
    });
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo-icon">📦</div>
          <div>
            <div className="brand-title">CAPACITYX</div>
            <div className="brand-subtitle">
              Smart Logistics Decision Agent &bull; Strands Decider 2B
              <span className="brand-badge">Tool Selection</span>
            </div>
          </div>
        </div>

        <div className="header-meta">
          <div className="status-pill" title={backendHealth.message}>
            <span
              className={`status-dot ${
                backendHealth.status === 'online' ? 'online' : 'simulated'
              }`}
            />
            <span>
              Backend:{' '}
              {backendHealth.status === 'online' ? `Connected (${backendPortDisplay})` : 'Offline / Standalone'}
            </span>
          </div>

          <div
            className="status-pill"
            title={
              modelStatus.is_real_model_available
                ? `Connected to Strands Decider server: ${modelStatus.checkpoint}`
                : `Model server offline at ${modelStatus.server_url || 'http://127.0.0.1:8099'}`
            }
          >
            <span
              className={`status-dot ${
                modelStatus.is_real_model_available ? 'online' : 'simulated'
              }`}
            />
            <span>
              Decider 2B:{' '}
              {modelStatus.is_real_model_available ? 'Online (Real Model)' : 'Offline (Safe Fallback)'}
            </span>
          </div>

          <button
            onClick={checkHealth}
            style={{
              background: 'transparent',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-secondary)',
              padding: '6px 12px',
              borderRadius: 'var(--radius-full)',
              cursor: 'pointer',
              fontSize: '12px',
            }}
            title={`Refresh backend and model status from ${BACKEND_BASE_URL}`}
          >
            ↻ Ping
          </button>
        </div>
      </header>

      {/* Quick Scenario Selector */}
      <section className="scenarios-bar">
        <div className="scenarios-header">
          <div className="scenarios-title">
            <span>⚡ Quick Scenarios (Pre-loaded Incident Presets)</span>
          </div>
          {activeScenarioId === 'custom' && (
            <span style={{ fontSize: '12px', color: 'var(--accent-cyan)' }}>
              ✏️ Custom Parameters Active
            </span>
          )}
        </div>

        <div className="scenarios-grid">
          {PRESET_SCENARIOS.map((sc) => (
            <button
              key={sc.id}
              className={`scenario-btn ${activeScenarioId === sc.id ? 'active' : ''}`}
              onClick={() => handleScenarioSelect(sc)}
            >
              <div className="scenario-name">
                <span>{sc.name}</span>
                {activeScenarioId === sc.id && <span>✓</span>}
              </div>
              <div className="scenario-desc">{sc.desc}</div>
            </button>
          ))}
        </div>
      </section>

      {/* Main 2-Column Dashboard Grid */}
      <main className="dashboard-grid">
        {/* Left Column: Shipment Incident Parameters Form */}
        <section className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">
              <span>📋</span> Shipment & Telemetry Input
            </h2>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Incident Parameters
            </span>
          </div>

          <form onSubmit={handleSimulateDecision}>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="shipment-id-input">Shipment ID</label>
                <input
                  id="shipment-id-input"
                  type="text"
                  className="form-control"
                  value={formData.shipmentId}
                  onChange={(e) => handleFieldChange('shipmentId', e.target.value)}
                  placeholder="e.g. SHP-8921-X"
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="delivery-deadline-input">Delivery Deadline</label>
                <input
                  id="delivery-deadline-input"
                  type="text"
                  className="form-control"
                  value={formData.deliveryDeadline}
                  onChange={(e) => handleFieldChange('deliveryDeadline', e.target.value)}
                  placeholder="e.g. 2026-10-20T14:30:00Z"
                  required
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="cargo-type-input">Cargo Classification</label>
                <input
                  id="cargo-type-input"
                  type="text"
                  className="form-control"
                  value={formData.cargoType}
                  onChange={(e) => handleFieldChange('cargoType', e.target.value)}
                  placeholder="e.g. Cold-Chain Perishables"
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="distance-km-input">Est. Distance Remaining (km)</label>
                <input
                  id="distance-km-input"
                  type="number"
                  className="form-control"
                  value={formData.estimatedDistanceKm}
                  onChange={(e) => handleFieldChange('estimatedDistanceKm', e.target.value)}
                  placeholder="e.g. 45"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="origin-input">Origin Depot</label>
                <input
                  id="origin-input"
                  type="text"
                  className="form-control"
                  value={formData.origin}
                  onChange={(e) => handleFieldChange('origin', e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="destination-input">Destination Center</label>
                <input
                  id="destination-input"
                  type="text"
                  className="form-control"
                  value={formData.destination}
                  onChange={(e) => handleFieldChange('destination', e.target.value)}
                />
              </div>
            </div>

            {/* Toggle 1: Truck Breakdown */}
            <div
              className={`toggle-tile ${formData.truckBreakdown ? 'checked-danger' : ''}`}
              onClick={() => handleFieldChange('truckBreakdown', !formData.truckBreakdown)}
            >
              <div className="toggle-tile-info">
                <div className="toggle-tile-title">
                  <span>🚨 Truck Breakdown Event</span>
                  {formData.truckBreakdown && (
                    <span style={{ fontSize: '11px', color: 'var(--accent-rose)', fontWeight: '700' }}>
                      [BREAKDOWN ACTIVE]
                    </span>
                  )}
                </div>
                <div className="toggle-tile-desc">
                  Primary vehicle immobilized or experiencing critical engine / transmission failure
                </div>
              </div>
              <label className="switch" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={formData.truckBreakdown}
                  onChange={(e) => handleFieldChange('truckBreakdown', e.target.checked)}
                />
                <span className="slider danger" />
              </label>
            </div>

            {formData.truckBreakdown && (
              <div className="form-group" style={{ marginBottom: '14px' }}>
                <label className="form-label" htmlFor="breakdown-severity-select">Breakdown Severity</label>
                <select
                  id="breakdown-severity-select"
                  className="form-control"
                  value={formData.breakdownSeverity}
                  onChange={(e) => handleFieldChange('breakdownSeverity', e.target.value)}
                >
                  <option value="LOW">LOW — Minor sensor issue or slow tire leak</option>
                  <option value="MODERATE">MODERATE — Limp mode or cooling fault</option>
                  <option value="CRITICAL">CRITICAL — Engine seizure, total immobilization</option>
                </select>
              </div>
            )}

            {/* Toggle 2: Replacement Vehicle Available */}
            <div
              className={`toggle-tile ${formData.replacementVehicleAvailable ? 'checked-positive' : ''}`}
              onClick={() =>
                handleFieldChange(
                  'replacementVehicleAvailable',
                  !formData.replacementVehicleAvailable
                )
              }
            >
              <div className="toggle-tile-info">
                <div className="toggle-tile-title">
                  <span>🚛 Replacement Vehicle Availability</span>
                  {formData.replacementVehicleAvailable ? (
                    <span style={{ fontSize: '11px', color: 'var(--accent-emerald)', fontWeight: '700' }}>
                      [AVAILABLE NEARBY]
                    </span>
                  ) : (
                    <span style={{ fontSize: '11px', color: 'var(--accent-amber)', fontWeight: '700' }}>
                      [NONE IN RANGE]
                    </span>
                  )}
                </div>
                <div className="toggle-tile-desc">
                  Reserve carrier or regional fleet backup unit accessible within 45 minutes
                </div>
              </div>
              <label className="switch" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={formData.replacementVehicleAvailable}
                  onChange={(e) =>
                    handleFieldChange('replacementVehicleAvailable', e.target.checked)
                  }
                />
                <span className="slider positive" />
              </label>
            </div>

            {/* Toggle 3: Customer Notified */}
            <div
              className={`toggle-tile ${formData.customerNotified ? 'checked-positive' : ''}`}
              onClick={() => handleFieldChange('customerNotified', !formData.customerNotified)}
            >
              <div className="toggle-tile-info">
                <div className="toggle-tile-title">
                  <span>📱 Customer Notified</span>
                  {formData.customerNotified ? (
                    <span style={{ fontSize: '11px', color: 'var(--accent-emerald)', fontWeight: '700' }}>
                      [ALREADY NOTIFIED]
                    </span>
                  ) : (
                    <span style={{ fontSize: '11px', color: 'var(--accent-amber)', fontWeight: '700' }}>
                      [PENDING ALERT]
                    </span>
                  )}
                </div>
                <div className="toggle-tile-desc">
                  Receiving client has been dispatched notice of transit status or incident
                </div>
              </div>
              <label className="switch" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={formData.customerNotified}
                  onChange={(e) => handleFieldChange('customerNotified', e.target.checked)}
                />
                <span className="slider positive" />
              </label>
            </div>

            <button
              id="simulate-decision-btn"
              type="submit"
              className="btn-submit"
              disabled={isLoading}
            >
              {isLoading ? (
                <>⏳ Evaluating Incident with Decider 2B...</>
              ) : (
                <>⚡ Evaluate Decision & Tools</>
              )}
            </button>
          </form>
        </section>

        {/* Right Column: Outcome Panel (Distinguishing Real Model vs Fallback) */}
        <section className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">
              <span>🎯</span> Decision Outcome & Tool Selection
            </h2>
            {decisionResult?.outcome ? (
              decisionResult.outcome.is_real_model ? (
                <span className="badge-action" style={{ borderColor: 'var(--accent-emerald)', color: 'var(--accent-emerald)' }}>
                  🤖 STRANDS DECIDER 2B REAL
                </span>
              ) : (
                <span className="badge-simulated">
                  🛡️ SAFE FALLBACK POLICY
                </span>
              )
            ) : (
              <span className="badge-simulated">
                AWAITING INPUT
              </span>
            )}
          </div>

          {!decisionResult ? (
            <div className="empty-state">
              <div className="empty-state-icon">🧭</div>
              <h3 style={{ fontSize: '16px', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Awaiting Incident Evaluation
              </h3>
              <p style={{ fontSize: '13px' }}>
                Select a quick scenario above or customize the incident fields, then click{' '}
                <strong style={{ color: 'var(--accent-cyan)' }}>"Evaluate Decision & Tools"</strong>.
              </p>
            </div>
          ) : (
            <div>
              {/* Engine Transparency Banner */}
              {decisionResult.outcome.is_real_model ? (
                <div className="outcome-banner" style={{ background: 'rgba(16, 185, 129, 0.1)', borderColor: 'rgba(16, 185, 129, 0.4)' }}>
                  <div className="outcome-banner-icon" style={{ color: 'var(--accent-emerald)' }}>🤖</div>
                  <div className="outcome-banner-content">
                    <h4 style={{ color: 'var(--accent-emerald)' }}>Strands Decider 2B Live Inference</h4>
                    <p>
                      Evaluated with real model <code>{decisionResult.outcome.model_id}</code>.
                      Calibrated confidence: <strong>{((decisionResult.outcome.model_confidence || 0) * 100).toFixed(1)}%</strong>.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="outcome-banner">
                  <div className="outcome-banner-icon">🛡️</div>
                  <div className="outcome-banner-content">
                    <h4>Deterministic Safe Fallback Engaged</h4>
                    <p>
                      <strong>{decisionResult.outcome.fallback_reason || 'Strands Decider 2B server is offline.'}</strong>
                      {' '}This decision was computed by deterministic logistics safety policies.
                    </p>
                  </div>
                </div>
              )}

              {/* Human Review Banner if required */}
              {decisionResult.outcome.human_review_required && (
                <div
                  style={{
                    background: 'rgba(239, 68, 68, 0.12)',
                    border: '1px solid rgba(239, 68, 68, 0.4)',
                    borderRadius: 'var(--radius-md)',
                    padding: '12px 14px',
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}
                >
                  <span style={{ fontSize: '20px' }}>⚠️</span>
                  <div>
                    <strong style={{ color: 'var(--accent-rose)', display: 'block', fontSize: '13px' }}>
                      Human Dispatcher Review Required
                    </strong>
                    <span style={{ fontSize: '12px', color: '#fca5a5' }}>
                      {decisionResult.outcome.is_real_model
                        ? `Confidence (${((decisionResult.outcome.model_confidence || 0) * 100).toFixed(1)}%) is below safety threshold, or risk condition requires dispatcher authorization.`
                        : 'Safe fallback engaged; automated dispatch is withheld pending human confirmation.'}
                    </span>
                  </div>
                </div>
              )}

              {/* Decision Summary Card */}
              <div
                className={`decision-card urgency-${(
                  decisionResult.outcome.urgency || 'low'
                ).toLowerCase()}`}
              >
                <div className="decision-meta">
                  <span className="badge-action">
                    Tool: {decisionResult.outcome.selected_action}
                  </span>
                  <span className={`badge-urgency ${decisionResult.outcome.urgency}`}>
                    Urgency: {decisionResult.outcome.urgency}
                  </span>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '11px',
                      color: decisionResult.outcome.is_real_model ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                    }}
                  >
                    Engine: {decisionResult.outcome.engine}
                  </span>
                </div>

                <h3 className="decision-headline">
                  {decisionResult.outcome.headline}
                </h3>

                <ul className="rationale-list">
                  {decisionResult.outcome.rationale.map((point, idx) => (
                    <li key={idx} className="rationale-item">
                      <span className="rationale-bullet">▸</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>

                {/* Key Metrics */}
                <div className="metrics-row">
                  <div className="metric-stat">
                    <div className="metric-label">Selected Tool</div>
                    <div className="metric-val" style={{ color: 'var(--accent-cyan)', fontSize: '14px' }}>
                      {decisionResult.outcome.selected_action}
                    </div>
                  </div>

                  <div className="metric-stat">
                    <div className="metric-label">Model Confidence</div>
                    <div
                      className="metric-val"
                      style={{
                        color:
                          decisionResult.outcome.model_confidence !== null
                            ? decisionResult.outcome.model_confidence >= 0.70
                              ? 'var(--accent-emerald)'
                              : 'var(--accent-amber)'
                            : 'var(--text-muted)',
                      }}
                    >
                      {decisionResult.outcome.model_confidence !== null
                        ? `${(decisionResult.outcome.model_confidence * 100).toFixed(1)}%`
                        : 'N/A (Fallback)'}
                    </div>
                  </div>

                  <div className="metric-stat">
                    <div className="metric-label">Estimated Delay</div>
                    <div
                      className="metric-val"
                      style={{
                        color:
                          decisionResult.outcome.estimated_delay_minutes > 60
                            ? 'var(--accent-rose)'
                            : decisionResult.outcome.estimated_delay_minutes > 0
                            ? 'var(--accent-amber)'
                            : 'var(--accent-emerald)',
                      }}
                    >
                      {decisionResult.outcome.estimated_delay_minutes > 0
                        ? `+${decisionResult.outcome.estimated_delay_minutes} min`
                        : 'On Time (0 min)'}
                    </div>
                  </div>

                  <div className="metric-stat">
                    <div className="metric-label">SLA Protection</div>
                    <div
                      className="metric-val"
                      style={{
                        color: decisionResult.outcome.deadline_met_projected
                          ? 'var(--accent-emerald)'
                          : 'var(--accent-rose)',
                      }}
                    >
                      {decisionResult.outcome.deadline_met_projected ? 'Achievable' : 'Breach Risk'}
                    </div>
                  </div>
                </div>

                {/* Probability Distribution of Alternative Tools (when real model inference provides them) */}
                {Object.keys(decisionResult.outcome.alternatives || {}).length > 0 && (
                  <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '8px' }}>
                      Model Tool Selection Probability Distribution
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {Object.entries(decisionResult.outcome.alternatives)
                        .sort((a, b) => b[1] - a[1])
                        .map(([toolName, prob]) => (
                          <div key={toolName} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                            <span style={{ width: '170px', fontFamily: 'var(--font-mono)', color: toolName === decisionResult.outcome.selected_action ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                              {toolName}
                            </span>
                            <div style={{ flex: 1, background: '#1e293b', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                              <div
                                style={{
                                  width: `${prob * 100}%`,
                                  height: '100%',
                                  background: toolName === decisionResult.outcome.selected_action ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.2)',
                                }}
                              />
                            </div>
                            <span style={{ width: '45px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Customer Communication Recommendation */}
              <div
                style={{
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  borderRadius: 'var(--radius-md)',
                  padding: '12px 14px',
                  marginBottom: '20px',
                  fontSize: '13px',
                }}
              >
                <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '2px' }}>
                  💬 Customer Advisory Status:
                </strong>
                <span style={{ color: '#cbd5e1' }}>
                  {decisionResult.outcome.customer_status_recommended}
                </span>
              </div>

              {/* Action State Breakdown Table */}
              <div className="comparison-section">
                <div className="comparison-header">
                  <span>⚖️ Action State Breakdown (Simulated vs Real)</span>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    Safety Guardrails Enforced
                  </span>
                </div>

                <div className="actions-table-wrapper">
                  <table className="actions-table">
                    <thead>
                      <tr>
                        <th>System / Subsystem</th>
                        <th>Planned Action</th>
                        <th>Real Cloud / Fleet Execution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {decisionResult.outcome.actions_preview?.map((act, index) => (
                        <tr key={index}>
                          <td style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                            {act.system}
                          </td>
                          <td>
                            <div style={{ marginBottom: '4px' }}>{act.action}</div>
                            <span className="badge-simulated">
                              {act.status}
                            </span>
                          </td>
                          <td>
                            <span className="badge-real-withheld">
                              🚫 Withheld (No Live Cloud Call)
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Developer JSON Inspector Toggle */}
              <div style={{ marginTop: '16px' }}>
                <button
                  type="button"
                  onClick={() => setShowJsonRaw(!showJsonRaw)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    fontSize: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  {showJsonRaw ? '▼ Hide Raw JSON Payload' : '▶ Inspect Raw Contract JSON'}
                </button>

                {showJsonRaw && (
                  <pre
                    style={{
                      marginTop: '8px',
                      padding: '12px',
                      background: '#05080f',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '11px',
                      fontFamily: 'var(--font-mono)',
                      color: '#a5f3fc',
                      overflowX: 'auto',
                    }}
                  >
                    {JSON.stringify(decisionResult, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          <strong>CAPACITYX</strong> — Smart Logistics Decision Agent &copy; 2026.
          {' '}&bull; Real Model: Strands Decider 2B &bull; Tools: 5 Typed Choices &bull; Fallback: Deterministic Policy
        </p>
      </footer>
    </div>
  );
}
