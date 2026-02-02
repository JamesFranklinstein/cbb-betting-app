import React, { useState, useEffect, Component } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import './styles.css';

// API base URL
const API_BASE = process.env.REACT_APP_API_URL || 'https://cbb-backend-production-9663.up.railway.app';

// Error Boundary Component to catch React errors gracefully
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ error, errorInfo });
    // Log error to console for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary-content">
            <h2>Something went wrong</h2>
            <p>The application encountered an unexpected error.</p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null, errorInfo: null });
                window.location.reload();
              }}
              className="error-boundary-button"
            >
              Reload Page
            </button>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="error-details">
                <summary>Error Details</summary>
                <pre>{this.state.error.toString()}</pre>
                <pre>{this.state.errorInfo?.componentStack}</pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main App Component
function App() {
  const [games, setGames] = useState([]);
  const [valueBets, setValueBets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('value-bets');
  const [minEdge, setMinEdge] = useState(0.03);
  const [betHistory, setBetHistory] = useState([]);
  const [betStats, setBetStats] = useState(null);

  useEffect(() => {
    fetchData();
  }, [minEdge]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [gamesRes, valueBetsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/games/today`),
        axios.get(`${API_BASE}/api/value-bets?min_edge=${minEdge}`)
      ]);
      setGames(gamesRes.data);
      setValueBets(valueBetsRes.data);

      // Auto-store today's value bets
      try {
        await axios.post(`${API_BASE}/api/bet-history/store?min_edge=${minEdge}`);
      } catch {
        // Storage endpoint may fail silently
      }

      // Fetch and update results for completed games (before loading history)
      try {
        await axios.post(`${API_BASE}/api/bet-history/fetch-results`);
      } catch {
        // Results fetch may fail silently
      }

      // Fetch bet history and stats
      try {
        const [historyRes, statsRes] = await Promise.all([
          axios.get(`${API_BASE}/api/bet-history?days=2`),
          axios.get(`${API_BASE}/api/bet-history/stats`)
        ]);
        setBetHistory(historyRes.data);
        setBetStats(statsRes.data);
      } catch {
        // History endpoints may not exist yet
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />
      
      <main className="main-content">
        <div className="container">
          {/* Tab Navigation */}
          <nav className="tabs">
            <button
              className={activeTab === 'value-bets' ? 'active' : ''}
              onClick={() => setActiveTab('value-bets')}
            >
              🎯 Value Bets ({valueBets.length})
            </button>
            <button
              className={activeTab === 'all-games' ? 'active' : ''}
              onClick={() => setActiveTab('all-games')}
            >
              📊 All Games ({games.length})
            </button>
            <button
              className={activeTab === 'stat-diffs' ? 'active' : ''}
              onClick={() => setActiveTab('stat-diffs')}
            >
              📋 Stat Diffs
            </button>
            <button
              className={activeTab === 'spreads' ? 'active' : ''}
              onClick={() => setActiveTab('spreads')}
            >
              📈 Spread Diffs
            </button>
            <button
              className={activeTab === 'totals' ? 'active' : ''}
              onClick={() => setActiveTab('totals')}
            >
              📉 Total Diffs
            </button>
            <button
              className={activeTab === 'performance' ? 'active' : ''}
              onClick={() => setActiveTab('performance')}
            >
              📈 Results {betHistory.length > 0 && `(${betHistory.length})`}
            </button>
          </nav>

          {/* Filters */}
          <div className="filters">
            <label>
              Min Edge: 
              <select value={minEdge} onChange={(e) => setMinEdge(parseFloat(e.target.value))}>
                <option value={0.02}>2%</option>
                <option value={0.03}>3%</option>
                <option value={0.05}>5%</option>
                <option value={0.07}>7%</option>
                <option value={0.10}>10%</option>
              </select>
            </label>
            <button onClick={fetchData} className="refresh-btn">
              🔄 Refresh
            </button>
          </div>

          {/* Content */}
          {loading && <LoadingSpinner />}
          {error && <ErrorMessage message={error} />}
          
          {!loading && !error && (
            <>
              {activeTab === 'value-bets' && (
                <ValueBetsList valueBets={valueBets} />
              )}
              {activeTab === 'all-games' && <GamesList games={games} />}
              {activeTab === 'stat-diffs' && <StatDiffsView games={games} />}
              {activeTab === 'spreads' && <SpreadComparison games={games} />}
              {activeTab === 'totals' && <TotalComparison games={games} />}
              {activeTab === 'performance' && (
                <ResultsDashboard
                  betHistory={betHistory}
                  betStats={betStats}
                />
              )}
            </>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}

// Header Component
function Header() {
  return (
    <header className="header">
      <div className="container">
        <h1>🏀 CBB Betting Dashboard</h1>
        <p className="date">{format(new Date(), 'EEEE, MMMM d, yyyy')}</p>
      </div>
    </header>
  );
}

// Footer Component
function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <p>Data from KenPom & The Odds API • Not financial advice • Gamble responsibly</p>
      </div>
    </footer>
  );
}

// Loading Spinner
function LoadingSpinner() {
  return (
    <div className="loading">
      <div className="spinner"></div>
      <p>Loading today's games...</p>
    </div>
  );
}

// Error Message
function ErrorMessage({ message }) {
  return (
    <div className="error">
      <p>⚠️ {message}</p>
      <p className="hint">Make sure the API server is running and API keys are configured.</p>
    </div>
  );
}

// Metrics Legend Component
function MetricsLegend() {
  const [showLegend, setShowLegend] = useState(false);

  return (
    <div className="metrics-legend-wrapper">
      <button
        className="legend-toggle"
        onClick={() => setShowLegend(!showLegend)}
      >
        {showLegend ? '▼' : '▶'} How to Read These Cards
      </button>

      {showLegend && (
        <div className="metrics-legend">
          <div className="legend-section">
            <h4>🎯 Key Betting Metrics</h4>
            <div className="legend-grid">
              <div className="legend-card">
                <div className="legend-header edge">Edge</div>
                <div className="legend-content">
                  <p><strong>What it means:</strong> Your predicted probability minus what the odds imply.</p>
                  <p><strong>Example:</strong> If you think a bet wins 55% of the time but Vegas odds imply 50%, you have a 5% edge.</p>
                  <p><strong>Good value:</strong> 3%+ is worth considering, 5%+ is solid, 7%+ is excellent</p>
                </div>
              </div>
              <div className="legend-card">
                <div className="legend-header ev">Expected Value (EV)</div>
                <div className="legend-content">
                  <p><strong>What it means:</strong> Average profit per dollar wagered over many bets.</p>
                  <p><strong>Example:</strong> +8% EV means you'd expect to profit $8 for every $100 wagered long-term.</p>
                  <p><strong>Good value:</strong> Any positive EV is good. Higher = more profitable long-term.</p>
                </div>
              </div>
              <div className="legend-card">
                <div className="legend-header kelly">Kelly Size</div>
                <div className="legend-content">
                  <p><strong>What it means:</strong> Mathematically optimal bet size as % of your bankroll.</p>
                  <p><strong>Note:</strong> We use 1/4 Kelly for safety (reduces variance).</p>
                  <p><strong>Usage:</strong> If Kelly shows 2% and you have $1000 bankroll, bet up to $20.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="legend-section">
            <h4>📊 Model vs Market Comparison</h4>
            <div className="legend-explanation">
              <p><strong>KenPom:</strong> Our model's prediction based on advanced basketball analytics (efficiency, tempo, four factors)</p>
              <p><strong>Vegas:</strong> The betting market's current line</p>
              <p><strong>Gap:</strong> The difference between KenPom and Vegas. Larger gaps often indicate value opportunities.</p>
              <ul className="gap-examples">
                <li><strong>Spread Gap +3.0:</strong> KenPom thinks the home team is 3 points better than Vegas does</li>
                <li><strong>Total Gap -4.0:</strong> KenPom projects 4 fewer points scored than the Vegas total</li>
              </ul>
            </div>
          </div>

          <div className="legend-section">
            <h4>🔥 Confidence Levels</h4>
            <div className="confidence-legend">
              <div className="conf-item high">
                <span className="conf-badge">HIGH</span>
                <span>Large edge (7%+) with strong statistical support. Best opportunities.</span>
              </div>
              <div className="conf-item medium">
                <span className="conf-badge">MEDIUM</span>
                <span>Solid edge (4-7%) worth considering. Good risk/reward balance.</span>
              </div>
              <div className="conf-item low">
                <span className="conf-badge">LOW</span>
                <span>Smaller edge (3-4%). May still be +EV but higher variance.</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Value Bets Summary Dashboard
function ValueBetsSummary({ valueBets }) {
  const highConf = valueBets.filter(b => b.bet.confidence === 'high').length;
  const medConf = valueBets.filter(b => b.bet.confidence === 'medium').length;
  const lowConf = valueBets.filter(b => b.bet.confidence === 'low').length;

  const spreadBets = valueBets.filter(b => b.bet.type === 'spread').length;
  const totalBets = valueBets.filter(b => b.bet.type === 'total').length;
  const mlBets = valueBets.filter(b => b.bet.type === 'moneyline').length;

  const uniqueGames = new Set(valueBets.map(b => `${b.home_team}-${b.away_team}`)).size;

  return (
    <div className="summary-dashboard">
      <div className="summary-header">
        <h3>Today's Summary</h3>
        <span className="summary-subtitle">Quick overview of identified opportunities</span>
      </div>
      <div className="summary-grid">
        <div className="summary-card total-bets">
          <span className="summary-value">{valueBets.length}</span>
          <span className="summary-label">Value Bets Found</span>
          <span className="summary-detail">across {uniqueGames} games</span>
        </div>
        <div className="summary-card confidence-breakdown">
          <span className="summary-label">By Confidence</span>
          <div className="confidence-bars">
            <div className="conf-bar high">
              <span className="conf-count">{highConf}</span>
              <span className="conf-label">High</span>
            </div>
            <div className="conf-bar medium">
              <span className="conf-count">{medConf}</span>
              <span className="conf-label">Medium</span>
            </div>
            <div className="conf-bar low">
              <span className="conf-count">{lowConf}</span>
              <span className="conf-label">Low</span>
            </div>
          </div>
        </div>
        <div className="summary-card bet-types">
          <span className="summary-label">Bet Types</span>
          <div className="type-breakdown">
            {spreadBets > 0 && <span className="type-item">📊 {spreadBets} Spread</span>}
            {totalBets > 0 && <span className="type-item">📈 {totalBets} Total</span>}
            {mlBets > 0 && <span className="type-item">💰 {mlBets} ML</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

// Value Bets List
function ValueBetsList({ valueBets }) {
  if (valueBets.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🔍</div>
        <h3>No Value Bets Found</h3>
        <p>No bets meeting your minimum edge criteria were identified for today's games.</p>
        <div className="empty-suggestions">
          <p><strong>Suggestions:</strong></p>
          <ul>
            <li>Try lowering the minimum edge filter</li>
            <li>Check back closer to game time when lines are more established</li>
            <li>View "All Games" tab to see today's matchups</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="value-bets-list">
      <div className="value-bets-header">
        <h2>Today's Value Bets</h2>
        <p className="value-bets-intro">
          These bets have been identified as having positive expected value based on KenPom's
          advanced analytics compared to current market odds.
        </p>
      </div>
      <ValueBetsSummary valueBets={valueBets} />
      <MetricsLegend />
      <div className="bets-grid">
        {valueBets.map((bet, idx) => (
          <ValueBetCard key={idx} bet={bet} />
        ))}
      </div>
    </div>
  );
}

// Value Bet Card
function ValueBetCard({ bet }) {
  const [expanded, setExpanded] = useState(false);
  const [showConfidenceBreakdown, setShowConfidenceBreakdown] = useState(false);

  const getTypeIcon = (type) => {
    switch(type) {
      case 'spread': return '📊';
      case 'total': return '📈';
      case 'moneyline': return '💰';
      default: return '🎯';
    }
  };

  const getConfidenceClass = (confidence) => {
    switch(confidence) {
      case 'high': return 'confidence-high';
      case 'medium': return 'confidence-medium';
      default: return 'confidence-low';
    }
  };

  const formatOdds = (odds) => {
    return odds > 0 ? `+${odds}` : odds;
  };

  // Generate explanation for why we recommend this bet
  const getValueExplanation = () => {
    const betType = bet.bet.type;
    const reasons = [];

    // Primary reason: High win probability from model
    const winProb = bet.bet.model_prob * 100;
    if (winProb >= 65) {
      reasons.push(`Model predicts ${winProb.toFixed(0)}% chance of winning - very high confidence`);
    } else if (winProb >= 60) {
      reasons.push(`Model predicts ${winProb.toFixed(0)}% chance of winning - high confidence`);
    } else if (winProb >= 55) {
      reasons.push(`Model predicts ${winProb.toFixed(0)}% chance of winning - good confidence`);
    }

    if (betType === 'spread') {
      const spreadDiff = Math.abs(bet.spread_diff);
      if (spreadDiff >= 2) {
        const favoredTeam = bet.spread_diff < 0 ? bet.home_team : bet.away_team;
        reasons.push(`KenPom projects ${favoredTeam} to be ${spreadDiff.toFixed(1)} points better than Vegas`);
      }
    } else if (betType === 'total') {
      const totalDiff = Math.abs(bet.total_diff);
      if (totalDiff >= 2) {
        const direction = bet.total_diff > 0 ? 'higher' : 'lower';
        reasons.push(`KenPom predicts the total ${totalDiff.toFixed(1)} points ${direction} than Vegas line`);
      }
    }

    if (bet.stat_summary) {
      const edges = [];
      if (bet.stat_summary.efficiency_edge) edges.push('efficiency');
      if (bet.stat_summary.shooting_edge) edges.push('shooting');
      if (bet.stat_summary.rebounding_edge) edges.push('rebounding');
      if (edges.length > 0) {
        reasons.push(`Statistical advantages in ${edges.join(', ')}`);
      }
    }

    return reasons;
  };

  const valueReasons = getValueExplanation();

  // Check for high severity data warnings
  const hasHighSeverityWarning = bet.data_warnings?.some(w => w.severity === 'high');

  return (
    <div className={`value-bet-card ${getConfidenceClass(bet.bet.confidence)} ${hasHighSeverityWarning ? 'has-warning' : ''}`}>
      {/* Data Quality Warning Banner */}
      {bet.data_warnings && bet.data_warnings.length > 0 && (
        <div className={`data-warning-banner ${hasHighSeverityWarning ? 'high' : 'medium'}`}>
          <span className="warning-icon">⚠️</span>
          <div className="warning-content">
            <span className="warning-title">Data Quality Warning</span>
            <span className="warning-message">
              {bet.data_warnings[0].message}
            </span>
          </div>
        </div>
      )}

      {/* Header with matchup and time */}
      <div className="bet-header">
        <span className="game-teams">
          <span className="rank">#{bet.away_rank}</span> {bet.away_team}
          <span className="at-symbol">@</span>
          <span className="rank">#{bet.home_rank}</span> {bet.home_team}
        </span>
        <span className="game-time">
          {format(new Date(bet.game_time), 'h:mm a')}
        </span>
      </div>

      {/* The Bet - Prominent Display */}
      <div className="the-bet-section">
        <div className="bet-type-badge">
          {getTypeIcon(bet.bet.type)} {bet.bet.type.toUpperCase()} BET
        </div>
        <div className="bet-recommendation">
          <div className="bet-pick">
            <strong>{bet.bet.side}</strong>
            {bet.bet.line != null && (
              <span className="bet-line"> {bet.bet.line > 0 ? '+' : ''}{bet.bet.line}</span>
            )}
          </div>
          <div className="bet-odds-book">
            <span className="odds">{formatOdds(bet.bet.odds)}</span>
            <span className="book">@ {bet.bet.book}</span>
          </div>
        </div>
      </div>

      {/* Line Comparison - Clearer Layout */}
      <div className="line-comparison-section">
        <div className="comparison-header">Model vs Market</div>
        <div className="comparison-grid">
          {/* Spread Comparison */}
          <div className="comparison-row">
            <span className="comparison-label">Spread</span>
            <div className="comparison-values">
              <div className="comparison-item kenpom">
                <span className="source">KenPom</span>
                <span className="value">{bet.home_team.split(' ').pop()} {bet.kenpom_spread > 0 ? '+' : ''}{bet.kenpom_spread?.toFixed(1)}</span>
              </div>
              <div className="comparison-item vegas">
                <span className="source">Vegas</span>
                <span className="value">{bet.home_team.split(' ').pop()} {bet.vegas_spread > 0 ? '+' : ''}{bet.vegas_spread?.toFixed(1)}</span>
              </div>
              <div className={`comparison-item diff ${Math.abs(bet.spread_diff) >= 2 ? 'significant' : ''}`}>
                <span className="source">Gap</span>
                <span className="value">{bet.spread_diff > 0 ? '+' : ''}{bet.spread_diff?.toFixed(1)} pts</span>
              </div>
            </div>
          </div>
          {/* Total Comparison */}
          <div className="comparison-row">
            <span className="comparison-label">Total</span>
            <div className="comparison-values">
              <div className="comparison-item kenpom">
                <span className="source">KenPom</span>
                <span className="value">{bet.kenpom_total?.toFixed(1)}</span>
              </div>
              <div className="comparison-item vegas">
                <span className="source">Vegas</span>
                <span className="value">{bet.vegas_total?.toFixed(1)}</span>
              </div>
              <div className={`comparison-item diff ${Math.abs(bet.total_diff) >= 3 ? 'significant' : ''}`}>
                <span className="source">Gap</span>
                <span className="value">{bet.total_diff > 0 ? '+' : ''}{bet.total_diff?.toFixed(1)} pts</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics with Explanations */}
      <div className="bet-metrics-enhanced">
        <div className="metric-card win-prob-metric">
          <div className="metric-header">
            <span className="metric-icon">🎯</span>
            <span className="metric-name">Win Probability</span>
          </div>
          <span className="metric-value">{(bet.bet.model_prob * 100).toFixed(1)}%</span>
          <span className="metric-explanation">Model's predicted chance this bet wins</span>
        </div>
        <div className="metric-card edge-metric">
          <div className="metric-header">
            <span className="metric-icon">📊</span>
            <span className="metric-name">Edge</span>
          </div>
          <span className="metric-value">{(bet.bet.edge * 100).toFixed(1)}%</span>
          <span className="metric-explanation">Model prob minus market implied prob</span>
        </div>
        <div className="metric-card ev-metric">
          <div className="metric-header">
            <span className="metric-icon">💵</span>
            <span className="metric-name">EV</span>
          </div>
          <span className="metric-value">{bet.bet.ev >= 0 ? '+' : ''}{(bet.bet.ev * 100).toFixed(1)}%</span>
          <span className="metric-explanation">Expected return per dollar wagered</span>
        </div>
      </div>

      {/* Why We Like This Bet */}
      {valueReasons.length > 0 && (
        <div className="value-explanation">
          <div className="explanation-header">
            <span className="header-icon">💡</span> Why We Like This Bet
          </div>
          <ul className="explanation-list">
            {valueReasons.map((reason, idx) => (
              <li key={idx}>{reason}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Expandable Stats Section */}
      <button
        className="expand-stats-btn"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? '▼ Hide Details' : '▶ Show Stat Breakdown'}
      </button>

      {expanded && (
        <div className="expanded-stats">
          {/* Stat Edges Summary */}
          {bet.stat_summary && (
            <div className="stat-edges-summary">
              <div className="edges-header">Team Advantages</div>
              <div className="edges-grid">
                {bet.stat_summary.efficiency_edge && (
                  <div className={`edge-badge ${bet.stat_summary.efficiency_edge}`}>
                    <span className="edge-category">Efficiency</span>
                    <span className="edge-team">
                      {bet.stat_summary.efficiency_edge === 'home' ? bet.home_team : bet.away_team}
                    </span>
                  </div>
                )}
                {bet.stat_summary.shooting_edge && (
                  <div className={`edge-badge ${bet.stat_summary.shooting_edge}`}>
                    <span className="edge-category">Shooting</span>
                    <span className="edge-team">
                      {bet.stat_summary.shooting_edge === 'home' ? bet.home_team : bet.away_team}
                    </span>
                  </div>
                )}
                {bet.stat_summary.rebounding_edge && (
                  <div className={`edge-badge ${bet.stat_summary.rebounding_edge}`}>
                    <span className="edge-category">Rebounding</span>
                    <span className="edge-team">
                      {bet.stat_summary.rebounding_edge === 'home' ? bet.home_team : bet.away_team}
                    </span>
                  </div>
                )}
                {bet.stat_summary.turnover_edge && (
                  <div className={`edge-badge ${bet.stat_summary.turnover_edge}`}>
                    <span className="edge-category">Ball Security</span>
                    <span className="edge-team">
                      {bet.stat_summary.turnover_edge === 'home' ? bet.home_team : bet.away_team}
                    </span>
                  </div>
                )}
                {bet.stat_summary.tempo_mismatch && (
                  <div className="edge-badge tempo">
                    <span className="edge-category">Pace</span>
                    <span className="edge-team">Tempo Mismatch</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Major Stat Differences */}
          {bet.major_stat_diffs && bet.major_stat_diffs.length > 0 && (
            <div className="major-stats-detailed">
              <div className="stats-header">Key Statistical Mismatches</div>
              <div className="stats-table">
                <div className="stats-table-header">
                  <span>Stat</span>
                  <span>{bet.away_team.split(' ').pop()}</span>
                  <span>{bet.home_team.split(' ').pop()}</span>
                  <span>Edge</span>
                </div>
                {bet.major_stat_diffs.map((diff, idx) => (
                  <div key={idx} className={`stats-table-row ${diff.advantage}`}>
                    <span className="stat-name">{diff.display_name}</span>
                    <span className={diff.advantage === 'away' ? 'advantage' : ''}>{diff.away_value.toFixed(1)}</span>
                    <span className={diff.advantage === 'home' ? 'advantage' : ''}>{diff.home_value.toFixed(1)}</span>
                    <span className="stat-edge">
                      {diff.advantage === 'home' ? bet.home_team.split(' ').pop() :
                       diff.advantage === 'away' ? bet.away_team.split(' ').pop() :
                       diff.stat_name === 'adj_tempo' ? 'Mismatch' : 'Even'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Confidence Badge with Score */}
      <div className="confidence-section">
        <div className={`confidence-badge ${bet.bet.confidence}`}>
          <span className="confidence-icon">
            {bet.bet.confidence === 'high' ? '🔥' : bet.bet.confidence === 'medium' ? '✓' : '○'}
          </span>
          {bet.bet.confidence.toUpperCase()} CONFIDENCE
          {bet.bet.confidence_score != null && (
            <span className="confidence-score">({Math.round(bet.bet.confidence_score)})</span>
          )}
        </div>
        <div className="confidence-explanation">
          {bet.bet.confidence === 'high' && 'Large edge with strong statistical support'}
          {bet.bet.confidence === 'medium' && 'Solid edge worth considering'}
          {bet.bet.confidence === 'low' && 'Smaller edge, proceed with caution'}
        </div>

        {/* Confidence Breakdown Toggle */}
        {bet.bet.confidence_factors && (
          <button
            className="confidence-breakdown-toggle"
            onClick={() => setShowConfidenceBreakdown(!showConfidenceBreakdown)}
          >
            {showConfidenceBreakdown ? '▼ Hide Score Breakdown' : '▶ Show Score Breakdown'}
          </button>
        )}

        {/* Confidence Breakdown Detail */}
        {showConfidenceBreakdown && bet.bet.confidence_factors && (
          <ConfidenceBreakdown factors={bet.bet.confidence_factors} />
        )}
      </div>

    </div>
  );
}

// Confidence Breakdown Component
function ConfidenceBreakdown({ factors }) {
  const factorList = [
    { key: 'edge', label: 'Edge Quality', max: 30, color: '#10b981' },
    { key: 'statistical', label: 'Statistical Edge', max: 20, color: '#3b82f6' },
    { key: 'model_agreement', label: 'Model Agreement', max: 15, color: '#8b5cf6' },
    { key: 'situational', label: 'Situational Factors', max: 10, color: '#ec4899' },
    { key: 'variance', label: 'Variance Penalty', max: 10, color: '#ef4444', isPenalty: true }
  ];

  return (
    <div className="confidence-breakdown">
      <div className="breakdown-header">Score Breakdown (max 75)</div>
      <div className="breakdown-bars">
        {factorList.map(({ key, label, max, color, isPenalty }) => {
          const factor = factors[key];
          if (!factor) return null;

          const score = isPenalty ? Math.abs(factor.penalty || 0) : (factor.score || 0);
          const percentage = (score / max) * 100;

          return (
            <div key={key} className="breakdown-row">
              <div className="breakdown-label">
                <span className="label-text">{label}</span>
                <span className="label-score" style={{ color }}>
                  {isPenalty ? `-${score.toFixed(0)}` : score.toFixed(0)}/{max}
                </span>
              </div>
              <div className="breakdown-bar-container">
                <div
                  className={`breakdown-bar ${isPenalty ? 'penalty' : ''}`}
                  style={{
                    width: `${Math.min(percentage, 100)}%`,
                    backgroundColor: color
                  }}
                />
              </div>
              {factor.interpretation && (
                <div className="breakdown-detail">{factor.interpretation}</div>
              )}
              {factor.note && (
                <div className="breakdown-detail note">{factor.note}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Games List
function GamesList({ games }) {
  if (games.length === 0) {
    return (
      <div className="empty-state">
        <p>📅 No games scheduled for today</p>
      </div>
    );
  }

  return (
    <div className="games-list">
      <h2>All Games Today</h2>
      <table className="games-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Matchup</th>
            <th>KenPom Spread</th>
            <th>Vegas Spread</th>
            <th>KenPom Total</th>
            <th>Vegas Total</th>
            <th>Win Prob</th>
            <th>Stats</th>
            <th>Values</th>
          </tr>
        </thead>
        <tbody>
          {games.map((game, idx) => (
            <tr key={idx}>
              <td>{format(new Date(game.game_time), 'h:mm a')}</td>
              <td className="matchup">
                <span className="rank">#{game.away_rank}</span> {game.away_team}
                <span className="at">@</span>
                <span className="rank">#{game.home_rank}</span> {game.home_team}
              </td>
              <td>{game.home_team} {game.kenpom_spread > 0 ? '+' : ''}{game.kenpom_spread.toFixed(1)}</td>
              <td>{game.vegas_spread !== 0 ? `${game.home_team} ${game.vegas_spread > 0 ? '+' : ''}${game.vegas_spread.toFixed(1)}` : '-'}</td>
              <td>{game.kenpom_total.toFixed(1)}</td>
              <td>{game.vegas_total !== 0 ? game.vegas_total.toFixed(1) : '-'}</td>
              <td>{(game.kenpom_home_win_prob * 100).toFixed(0)}%</td>
              <td>
                {game.major_stat_diffs && game.major_stat_diffs.length > 0 ? (
                  <span className="stat-diff-count" title={game.major_stat_diffs.map(d => d.display_name).join(', ')}>
                    {game.major_stat_diffs.length}
                  </span>
                ) : (
                  <span className="no-values">-</span>
                )}
              </td>
              <td>
                {game.value_bets.length > 0 ? (
                  <span className="value-count">{game.value_bets.length}</span>
                ) : (
                  <span className="no-values">0</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Spread Comparison
function SpreadComparison({ games }) {
  const spreadGames = games
    .filter(g => g.vegas_spread !== 0 && Math.abs(g.spread_diff) >= 1.5)
    .sort((a, b) => Math.abs(b.spread_diff) - Math.abs(a.spread_diff));

  return (
    <div className="comparison">
      <h2>Spread Discrepancies</h2>
      <p className="subtitle">Games where KenPom differs from Vegas by 1.5+ points</p>
      
      {spreadGames.length === 0 ? (
        <div className="empty-state">
          <p>No significant spread differences found</p>
        </div>
      ) : (
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Game</th>
              <th>KenPom</th>
              <th>Vegas</th>
              <th>Diff</th>
              <th>Lean</th>
            </tr>
          </thead>
          <tbody>
            {spreadGames.map((game, idx) => (
              <tr key={idx}>
                <td>{game.away_team} @ {game.home_team}</td>
                <td>{game.home_team} {game.kenpom_spread > 0 ? '+' : ''}{game.kenpom_spread.toFixed(1)}</td>
                <td>{game.home_team} {game.vegas_spread > 0 ? '+' : ''}{game.vegas_spread.toFixed(1)}</td>
                <td className={Math.abs(game.spread_diff) >= 3 ? 'big-diff' : 'diff'}>
                  {game.spread_diff > 0 ? '+' : ''}{game.spread_diff.toFixed(1)}
                </td>
                <td className="lean">
                  {game.spread_diff > 0 ? game.home_team : game.away_team}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// Total Comparison
function TotalComparison({ games }) {
  const totalGames = games
    .filter(g => g.vegas_total !== 0 && Math.abs(g.total_diff) >= 2)
    .sort((a, b) => Math.abs(b.total_diff) - Math.abs(a.total_diff));

  return (
    <div className="comparison">
      <h2>Total Discrepancies</h2>
      <p className="subtitle">Games where KenPom total differs from Vegas by 2+ points</p>

      {totalGames.length === 0 ? (
        <div className="empty-state">
          <p>No significant total differences found</p>
        </div>
      ) : (
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Game</th>
              <th>KenPom</th>
              <th>Vegas</th>
              <th>Diff</th>
              <th>Lean</th>
            </tr>
          </thead>
          <tbody>
            {totalGames.map((game, idx) => (
              <tr key={idx}>
                <td>{game.away_team} @ {game.home_team}</td>
                <td>{game.kenpom_total.toFixed(1)}</td>
                <td>{game.vegas_total.toFixed(1)}</td>
                <td className={Math.abs(game.total_diff) >= 4 ? 'big-diff' : 'diff'}>
                  {game.total_diff.toFixed(1)}
                </td>
                <td className={game.total_diff > 0 ? 'lean-over' : 'lean-under'}>
                  {game.total_diff > 0 ? 'OVER' : 'UNDER'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// Statistical Differences View
function StatDiffsView({ games }) {
  const [selectedGame, setSelectedGame] = useState(null);

  // Filter games with major stat differences
  const gamesWithMajorDiffs = games.filter(g =>
    g.major_stat_diffs && g.major_stat_diffs.length > 0
  ).sort((a, b) => (b.major_stat_diffs?.length || 0) - (a.major_stat_diffs?.length || 0));

  return (
    <div className="stat-diffs-view">
      <h2>Major Statistical Differences</h2>
      <p className="subtitle">Games with significant statistical mismatches between teams</p>

      {gamesWithMajorDiffs.length === 0 ? (
        <div className="empty-state">
          <p>No games with major statistical differences found</p>
        </div>
      ) : (
        <div className="stat-diffs-container">
          {/* Game List */}
          <div className="stat-diffs-list">
            {gamesWithMajorDiffs.map((game, idx) => (
              <div
                key={idx}
                className={`stat-diff-game-card ${selectedGame === idx ? 'selected' : ''}`}
                onClick={() => setSelectedGame(selectedGame === idx ? null : idx)}
              >
                <div className="game-header">
                  <span className="game-matchup">
                    #{game.away_rank} {game.away_team} @ #{game.home_rank} {game.home_team}
                  </span>
                  <span className="major-count">
                    {game.major_stat_diffs.length} major diff{game.major_stat_diffs.length !== 1 ? 's' : ''}
                  </span>
                </div>
                <div className="major-diffs-summary">
                  {game.major_stat_diffs.slice(0, 3).map((diff, i) => (
                    <span key={i} className={`diff-badge ${diff.advantage}`}>
                      {diff.display_name}: {diff.advantage === 'home' ? game.home_team : game.away_team}
                    </span>
                  ))}
                  {game.major_stat_diffs.length > 3 && (
                    <span className="more-badge">+{game.major_stat_diffs.length - 3} more</span>
                  )}
                </div>

                {/* Expanded Detail View */}
                {selectedGame === idx && game.stat_comparison && (
                  <StatComparisonDetail game={game} />
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <StatDiffsLegend />
    </div>
  );
}

// Stat Comparison Detail Component
function StatComparisonDetail({ game }) {
  const comp = game.stat_comparison;
  if (!comp) return null;

  return (
    <div className="stat-comparison-detail" onClick={e => e.stopPropagation()}>
      {/* Summary Edges */}
      <div className="edge-summary">
        {comp.efficiency_edge && (
          <div className="edge-item">
            <span className="edge-label">Efficiency Edge:</span>
            <span className={`edge-value ${comp.efficiency_edge}`}>
              {comp.efficiency_edge === 'home' ? game.home_team : game.away_team}
            </span>
          </div>
        )}
        {comp.shooting_edge && (
          <div className="edge-item">
            <span className="edge-label">Shooting Edge:</span>
            <span className={`edge-value ${comp.shooting_edge}`}>
              {comp.shooting_edge === 'home' ? game.home_team : game.away_team}
            </span>
          </div>
        )}
        {comp.rebounding_edge && (
          <div className="edge-item">
            <span className="edge-label">Rebounding Edge:</span>
            <span className={`edge-value ${comp.rebounding_edge}`}>
              {comp.rebounding_edge === 'home' ? game.home_team : game.away_team}
            </span>
          </div>
        )}
        {comp.turnover_edge && (
          <div className="edge-item">
            <span className="edge-label">Turnover Edge:</span>
            <span className={`edge-value ${comp.turnover_edge}`}>
              {comp.turnover_edge === 'home' ? game.home_team : game.away_team}
            </span>
          </div>
        )}
        {comp.tempo_mismatch && (
          <div className="edge-item tempo-mismatch">
            <span className="edge-label">Tempo Mismatch:</span>
            <span className="edge-value">Yes</span>
          </div>
        )}
      </div>

      {/* Detailed Stats Table */}
      <table className="stats-detail-table">
        <thead>
          <tr>
            <th>Stat</th>
            <th>{game.away_team}</th>
            <th>{game.home_team}</th>
            <th>Diff</th>
            <th>Edge</th>
          </tr>
        </thead>
        <tbody>
          {comp.stat_differences
            .filter(d => d.significance !== 'minor')
            .map((diff, idx) => (
              <tr key={idx} className={diff.significance}>
                <td>{diff.display_name}</td>
                <td className={diff.advantage === 'away' ? 'advantage' : ''}>
                  {formatStatValue(diff.away_value, diff.stat_name)}
                </td>
                <td className={diff.advantage === 'home' ? 'advantage' : ''}>
                  {formatStatValue(diff.home_value, diff.stat_name)}
                </td>
                <td className={`diff-cell ${diff.significance}`}>
                  {diff.difference > 0 ? '+' : ''}{formatStatValue(diff.difference, diff.stat_name)}
                </td>
                <td className={`edge-cell ${diff.advantage}`}>
                  {diff.advantage === 'neutral'
                    ? (diff.stat_name === 'adj_tempo' ? 'Mismatch' : '-')
                    : diff.advantage === 'home'
                      ? game.home_team.split(' ').pop()
                      : game.away_team.split(' ').pop()}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

// Helper function to format stat values
function formatStatValue(value, statName) {
  if (statName.includes('pct') || statName.includes('rate')) {
    return value.toFixed(1) + '%';
  }
  return value.toFixed(1);
}

// Results Dashboard Component - Shows automatic bet results
function ResultsDashboard({ betHistory, betStats }) {
  const [filter, setFilter] = useState('all'); // all, pending, completed

  const filteredBets = betHistory.filter(bet => {
    if (filter === 'pending') return !bet.result;
    if (filter === 'completed') return bet.result;
    return true;
  });

  const completedBets = betHistory.filter(b => b.result);
  const pendingBets = betHistory.filter(b => !b.result);

  return (
    <div className="performance-dashboard">
      <div className="dashboard-header">
        <div className="header-row">
          <div>
            <h2>📈 Bet Results</h2>
            <p className="dashboard-intro">Results update automatically every hour.</p>
          </div>
        </div>
      </div>

      {/* Summary Cards from Backend Stats */}
      {betStats && (
        <div className="performance-summary">
          <div className="perf-card total">
            <div className="perf-value">{betStats.total_bets}</div>
            <div className="perf-label">Total Bets</div>
          </div>
          <div className="perf-card record">
            <div className="perf-value">
              {betStats.total_wins}-{betStats.total_losses}
              {betStats.total_pushes > 0 && `-${betStats.total_pushes}`}
            </div>
            <div className="perf-label">Record</div>
          </div>
          <div className="perf-card win-rate">
            <div className="perf-value">{(betStats.win_rate * 100).toFixed(1)}%</div>
            <div className="perf-label">Win Rate</div>
          </div>
          <div className="perf-card high-conf">
            <div className="perf-value">{(betStats.high_conf?.win_rate * 100 || 0).toFixed(1)}%</div>
            <div className="perf-label">High Conf Win%</div>
          </div>
        </div>
      )}

      {/* Detailed Stats Cards */}
      {betStats && (
        <div className="stats-breakdown">
          <div className="stats-card">
            <h4>By Confidence</h4>
            <div className="stat-rows">
              <div className="stat-row">
                <span className="tier high">High</span>
                <span className="record">{betStats.high_conf?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.high_conf?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="tier medium">Medium</span>
                <span className="record">{betStats.medium_conf?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.medium_conf?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="tier low">Low</span>
                <span className="record">{betStats.low_conf?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.low_conf?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          <div className="stats-card">
            <h4>By Bet Type</h4>
            <div className="stat-rows">
              <div className="stat-row">
                <span className="type">Spread</span>
                <span className="record">{betStats.by_type?.spread?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.by_type?.spread?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="type">Total</span>
                <span className="record">{betStats.by_type?.total?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.by_type?.total?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="type">ML</span>
                <span className="record">{betStats.by_type?.moneyline?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.by_type?.moneyline?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          <div className="stats-card">
            <h4>Recent Performance</h4>
            <div className="stat-rows">
              <div className="stat-row">
                <span className="period">Today</span>
                <span className="record">{betStats.recent?.today?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.recent?.today?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="period">Yesterday</span>
                <span className="record">{betStats.recent?.yesterday?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.recent?.yesterday?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="period">Last 7 Days</span>
                <span className="record">{betStats.recent?.last_7_days?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.recent?.last_7_days?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              {betStats.streaks?.current > 0 && (
                <div className="stat-row streak">
                  <span className="period">Current Streak</span>
                  <span className={`streak-value ${betStats.streaks.type === 'W' ? 'win' : 'loss'}`}>
                    {betStats.streaks.current}{betStats.streaks.type}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Filter Tabs */}
      <div className="performance-filters">
        <button
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}
        >
          All ({betHistory.length})
        </button>
        <button
          className={filter === 'pending' ? 'active' : ''}
          onClick={() => setFilter('pending')}
        >
          Pending ({pendingBets.length})
        </button>
        <button
          className={filter === 'completed' ? 'active' : ''}
          onClick={() => setFilter('completed')}
        >
          Completed ({completedBets.length})
        </button>
      </div>

      {/* Bet History List */}
      {filteredBets.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <h3>No Bet History</h3>
          <p>Value bets are automatically tracked when identified. Check back after games complete to see results.</p>
        </div>
      ) : (
        <div className="tracked-bets-list">
          {filteredBets.map(bet => (
            <HistoryBetCard key={bet.id} bet={bet} />
          ))}
        </div>
      )}
    </div>
  );
}

// History Bet Card - Read-only display of bet with automatic result
function HistoryBetCard({ bet }) {
  const getResultClass = (result) => {
    if (result === 'win') return 'result-win';
    if (result === 'loss') return 'result-loss';
    if (result === 'push') return 'result-push';
    return '';
  };

  const formatOdds = (odds) => {
    return odds > 0 ? `+${odds}` : odds;
  };

  return (
    <div className={`tracked-bet-card ${getResultClass(bet.result)}`}>
      <div className="tracked-header">
        <div className="tracked-matchup">
          <span className="tracked-teams">
            #{bet.away_rank} {bet.away_team} @ #{bet.home_rank} {bet.home_team}
          </span>
          <span className="tracked-date">
            {bet.date}
          </span>
        </div>
        {bet.result && (
          <div className={`result-badge-small ${bet.result}`}>
            {bet.result.toUpperCase()}
          </div>
        )}
      </div>

      <div className="tracked-bet-info">
        <div className="bet-detail">
          <span className="detail-type">{bet.bet.type.toUpperCase()}</span>
          <span className="detail-pick">
            {bet.bet.side}
            {bet.bet.line != null && ` ${bet.bet.line > 0 ? '+' : ''}${bet.bet.line}`}
          </span>
          <span className="detail-odds">{formatOdds(bet.bet.odds)}</span>
        </div>

        <div className="bet-metrics-small">
          <span className="metric win-prob">Win%: {(bet.bet.model_prob * 100).toFixed(0)}%</span>
          <span className="metric edge">Edge: {(bet.bet.edge * 100).toFixed(1)}%</span>
          <span className={`metric conf conf-${bet.bet.confidence}`}>
            {bet.bet.confidence.toUpperCase()}
            {bet.bet.confidence_score && ` (${Math.round(bet.bet.confidence_score)})`}
          </span>
        </div>
      </div>

      {/* Score Display for Completed Games */}
      {bet.result && bet.home_score !== null && (
        <div className="score-display">
          <span className="final-score">
            Final: {bet.away_team} {bet.away_score} - {bet.home_team} {bet.home_score}
          </span>
          {bet.bet.type === 'spread' && (
            <span className="actual-margin">
              Margin: {bet.actual_margin > 0 ? '+' : ''}{bet.actual_margin}
            </span>
          )}
          {bet.bet.type === 'total' && (
            <span className="actual-total">
              Total: {bet.actual_total}
            </span>
          )}
        </div>
      )}

      {/* Pending indicator */}
      {!bet.result && (
        <div className="pending-indicator">
          <span className="pending-icon">⏳</span> Awaiting Result
        </div>
      )}
    </div>
  );
}

// Stats Differences Legend
function StatDiffsLegend() {
  return (
    <div className="stats-legend">
      <h3>Understanding the Stats</h3>
      <div className="legend-grid">
        <div className="legend-section">
          <h4>Efficiency</h4>
          <div className="legend-item">
            <span className="stat-name">Adj EM</span>
            <span className="stat-desc">Adjusted Efficiency Margin - points scored vs allowed per 100 possessions</span>
          </div>
          <div className="legend-item">
            <span className="stat-name">Adj OE/DE</span>
            <span className="stat-desc">Offensive/Defensive efficiency - points per 100 possessions</span>
          </div>
        </div>
        <div className="legend-section">
          <h4>Four Factors (Offense)</h4>
          <div className="legend-item">
            <span className="stat-name">eFG%</span>
            <span className="stat-desc">Effective FG% - shooting accuracy accounting for 3s</span>
          </div>
          <div className="legend-item">
            <span className="stat-name">TO%</span>
            <span className="stat-desc">Turnover % - turnovers per 100 plays (lower is better)</span>
          </div>
          <div className="legend-item">
            <span className="stat-name">OR%</span>
            <span className="stat-desc">Offensive rebound % - second chance opportunities</span>
          </div>
          <div className="legend-item">
            <span className="stat-name">FT Rate</span>
            <span className="stat-desc">Free throw rate - FTAs per FGA</span>
          </div>
        </div>
        <div className="legend-section">
          <h4>Four Factors (Defense)</h4>
          <div className="legend-item">
            <span className="stat-name">Opp eFG%</span>
            <span className="stat-desc">Opponent eFG% - how well defense contests shots</span>
          </div>
          <div className="legend-item">
            <span className="stat-name">Forced TO%</span>
            <span className="stat-desc">Forced turnover % - creating turnovers (higher is better)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Wrap App with ErrorBoundary for graceful error handling
function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}

export default AppWithErrorBoundary;
