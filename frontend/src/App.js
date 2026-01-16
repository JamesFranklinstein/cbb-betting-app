import React, { useState, useEffect, Component } from 'react';
import axios from 'axios';
import { format } from 'date-fns';

// API base URL
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

      // Fetch bet history and stats
      try {
        const [historyRes, statsRes] = await Promise.all([
          axios.get(`${API_BASE}/api/bet-history?days=30`),
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
      
      <style>{styles}</style>
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

  // Generate explanation for why this bet has value
  const getValueExplanation = () => {
    const betType = bet.bet.type;
    const reasons = [];

    if (betType === 'spread') {
      const spreadDiff = Math.abs(bet.spread_diff);
      if (spreadDiff >= 2) {
        // spread_diff = kenpom_spread - vegas_spread
        // Negative spread = home team favored
        // If spread_diff < 0, KenPom has home team MORE favored than Vegas
        // If spread_diff > 0, KenPom has away team MORE favored than Vegas
        const favoredTeam = bet.spread_diff < 0 ? bet.home_team : bet.away_team;
        reasons.push(`KenPom projects ${favoredTeam} to be ${spreadDiff.toFixed(1)} points better than Vegas thinks`);
      }
    } else if (betType === 'total') {
      const totalDiff = Math.abs(bet.total_diff);
      if (totalDiff >= 2) {
        const direction = bet.total_diff > 0 ? 'higher' : 'lower';
        reasons.push(`KenPom predicts the total ${totalDiff.toFixed(1)} points ${direction} than Vegas line`);
      }
    }

    if (bet.bet.edge >= 0.05) {
      reasons.push(`Strong ${(bet.bet.edge * 100).toFixed(1)}% edge over the market`);
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

  return (
    <div className={`value-bet-card ${getConfidenceClass(bet.bet.confidence)}`}>
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
        <div className="metric-card edge-metric">
          <div className="metric-header">
            <span className="metric-icon">🎯</span>
            <span className="metric-name">Edge</span>
          </div>
          <span className="metric-value">{(bet.bet.edge * 100).toFixed(1)}%</span>
          <span className="metric-explanation">Your probability advantage over the market</span>
        </div>
        <div className="metric-card ev-metric">
          <div className="metric-header">
            <span className="metric-icon">💵</span>
            <span className="metric-name">Expected Value</span>
          </div>
          <span className="metric-value">{bet.bet.ev >= 0 ? '+' : ''}{(bet.bet.ev * 100).toFixed(1)}%</span>
          <span className="metric-explanation">Avg return per $1 wagered long-term</span>
        </div>
        <div className="metric-card kelly-metric">
          <div className="metric-header">
            <span className="metric-icon">📊</span>
            <span className="metric-name">Kelly Size</span>
          </div>
          <span className="metric-value">{(bet.bet.kelly * 100).toFixed(1)}%</span>
          <span className="metric-explanation">Optimal bankroll % to wager</span>
        </div>
      </div>

      {/* Why This Bet Has Value */}
      {valueReasons.length > 0 && (
        <div className="value-explanation">
          <div className="explanation-header">
            <span className="header-icon">💡</span> Why This Bet Has Value
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
                <span className="period">Last 7 Days</span>
                <span className="record">{betStats.recent?.last_7_days?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.recent?.last_7_days?.win_rate * 100 || 0).toFixed(0)}%</span>
              </div>
              <div className="stat-row">
                <span className="period">Last 30 Days</span>
                <span className="record">{betStats.recent?.last_30_days?.record || '0-0-0'}</span>
                <span className="rate">{(betStats.recent?.last_30_days?.win_rate * 100 || 0).toFixed(0)}%</span>
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
          <span className="metric edge">Edge: {(bet.bet.edge * 100).toFixed(1)}%</span>
          <span className="metric ev">EV: {(bet.bet.ev * 100).toFixed(1)}%</span>
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

// Styles
const styles = `
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
  }

  .header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 24px 0;
    border-bottom: 1px solid #334155;
  }

  .header h1 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .header .date {
    color: #94a3b8;
    font-size: 14px;
  }

  .main-content {
    padding: 24px 0;
    min-height: calc(100vh - 200px);
  }

  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  .tabs button {
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    padding: 10px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
  }

  .tabs button:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .tabs button.active {
    background: #3b82f6;
    border-color: #3b82f6;
    color: white;
  }

  .filters {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 24px;
    padding: 16px;
    background: #1e293b;
    border-radius: 8px;
  }

  .filters label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #94a3b8;
    font-size: 14px;
  }

  .filters select {
    background: #0f172a;
    border: 1px solid #334155;
    color: #e2e8f0;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
  }

  .refresh-btn {
    background: #10b981;
    border: none;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    margin-left: auto;
  }

  .refresh-btn:hover {
    background: #059669;
  }

  .loading {
    text-align: center;
    padding: 60px 20px;
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #334155;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 16px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error {
    background: #7f1d1d;
    border: 1px solid #991b1b;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
  }

  .error .hint {
    color: #fca5a5;
    font-size: 14px;
    margin-top: 8px;
  }

  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #94a3b8;
    background: #1e293b;
    border-radius: 12px;
    max-width: 500px;
    margin: 40px auto;
  }

  .empty-state .empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
  }

  .empty-state h3 {
    color: #e2e8f0;
    margin-bottom: 12px;
    font-size: 20px;
  }

  .empty-state p {
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .empty-suggestions {
    text-align: left;
    background: #0f172a;
    padding: 16px;
    border-radius: 8px;
  }

  .empty-suggestions p {
    margin-bottom: 10px;
    color: #e2e8f0;
  }

  .empty-suggestions ul {
    margin: 0;
    padding-left: 20px;
  }

  .empty-suggestions li {
    margin-bottom: 8px;
    font-size: 14px;
  }

  .empty-state .hint {
    font-size: 14px;
    margin-top: 8px;
  }

  /* Value Bets Header */
  .value-bets-header {
    margin-bottom: 20px;
  }

  .value-bets-header h2 {
    margin-bottom: 8px;
  }

  .value-bets-intro {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.5;
  }

  /* Summary Dashboard */
  .summary-dashboard {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #334155;
  }

  .summary-header {
    margin-bottom: 16px;
  }

  .summary-header h3 {
    font-size: 16px;
    color: #e2e8f0;
    margin-bottom: 4px;
  }

  .summary-subtitle {
    font-size: 13px;
    color: #64748b;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
  }

  .summary-card {
    background: #0f172a;
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    border: 1px solid #334155;
  }

  .summary-card.total-bets {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-color: #3b82f680;
  }

  .summary-value {
    font-size: 36px;
    font-weight: 700;
    color: #3b82f6;
    line-height: 1;
  }

  .summary-label {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .summary-detail {
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
  }

  .confidence-bars {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }

  .conf-bar {
    flex: 1;
    text-align: center;
    padding: 8px 4px;
    border-radius: 6px;
  }

  .conf-bar.high {
    background: #065f46;
  }

  .conf-bar.medium {
    background: #78350f;
  }

  .conf-bar.low {
    background: #334155;
  }

  .conf-count {
    display: block;
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
  }

  .conf-bar.high .conf-count { color: #6ee7b7; }
  .conf-bar.medium .conf-count { color: #fcd34d; }
  .conf-bar.low .conf-count { color: #94a3b8; }

  .conf-bar .conf-label {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
  }

  .type-breakdown {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
  }

  .type-item {
    font-size: 13px;
    color: #e2e8f0;
  }

  .avg-metrics {
    display: flex;
    gap: 16px;
    margin-top: 8px;
  }

  .avg-item {
    flex: 1;
    text-align: center;
  }

  .avg-value {
    display: block;
    font-size: 20px;
    font-weight: 700;
  }

  .avg-value.edge { color: #10b981; }
  .avg-value.ev { color: #3b82f6; }

  .avg-label {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
  }

  h2 {
    font-size: 20px;
    margin-bottom: 16px;
  }

  .subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 16px;
    margin-top: -8px;
  }

  /* Enhanced Metrics Legend */
  .metrics-legend-wrapper {
    margin-bottom: 20px;
  }

  .legend-toggle {
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    padding: 12px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    width: 100%;
    text-align: left;
    transition: all 0.2s;
  }

  .legend-toggle:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .metrics-legend {
    background: #1e293b;
    border-radius: 0 0 8px 8px;
    padding: 20px;
    border: 1px solid #334155;
    border-top: none;
    margin-top: -1px;
  }

  .metrics-legend .legend-section {
    margin-bottom: 24px;
  }

  .metrics-legend .legend-section:last-child {
    margin-bottom: 0;
  }

  .metrics-legend h4 {
    font-size: 15px;
    color: #e2e8f0;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #334155;
  }

  .metrics-legend .legend-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  .legend-card {
    background: #0f172a;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid #334155;
  }

  .legend-card .legend-header {
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 2px solid;
  }

  .legend-card .legend-header.edge {
    color: #10b981;
    border-color: #10b981;
  }

  .legend-card .legend-header.ev {
    color: #3b82f6;
    border-color: #3b82f6;
  }

  .legend-card .legend-header.kelly {
    color: #f59e0b;
    border-color: #f59e0b;
  }

  .legend-card .legend-content p {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 8px;
    line-height: 1.5;
  }

  .legend-card .legend-content p:last-child {
    margin-bottom: 0;
  }

  .legend-card .legend-content strong {
    color: #e2e8f0;
  }

  .legend-explanation {
    background: #0f172a;
    border-radius: 8px;
    padding: 16px;
  }

  .legend-explanation p {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 10px;
    line-height: 1.5;
  }

  .legend-explanation strong {
    color: #e2e8f0;
  }

  .gap-examples {
    list-style: none;
    margin-top: 12px;
    padding-left: 0;
  }

  .gap-examples li {
    font-size: 12px;
    color: #94a3b8;
    padding: 8px 12px;
    background: #1e293b;
    border-radius: 6px;
    margin-bottom: 6px;
  }

  .gap-examples li strong {
    color: #10b981;
  }

  .confidence-legend {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .conf-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: #0f172a;
    border-radius: 6px;
  }

  .conf-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 4px;
    min-width: 70px;
    text-align: center;
  }

  .conf-item.high .conf-badge {
    background: #065f46;
    color: #6ee7b7;
  }

  .conf-item.medium .conf-badge {
    background: #78350f;
    color: #fcd34d;
  }

  .conf-item.low .conf-badge {
    background: #334155;
    color: #94a3b8;
  }

  .conf-item span:last-child {
    font-size: 13px;
    color: #94a3b8;
  }

  /* Value Bets Grid */
  .bets-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }

  .value-bet-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 20px;
    border-left: 4px solid #64748b;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .value-bet-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  }

  .value-bet-card.confidence-high {
    border-left-color: #10b981;
    background: linear-gradient(135deg, #1e293b 0%, #0f2920 100%);
  }

  .value-bet-card.confidence-medium {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #1e293b 0%, #2a2010 100%);
  }

  .value-bet-card.confidence-low {
    border-left-color: #64748b;
  }

  .bet-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #334155;
  }

  .game-teams {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 15px;
    line-height: 1.4;
  }

  .game-teams .rank {
    color: #64748b;
    font-size: 12px;
    font-weight: 500;
  }

  .game-teams .at-symbol {
    color: #64748b;
    margin: 0 6px;
  }

  .game-time {
    color: #94a3b8;
    font-size: 13px;
    background: #0f172a;
    padding: 4px 10px;
    border-radius: 6px;
  }

  /* The Bet Section */
  .the-bet-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid #334155;
  }

  .bet-type-badge {
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }

  .bet-recommendation {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .bet-pick {
    font-size: 22px;
    color: #e2e8f0;
  }

  .bet-pick strong {
    color: #fff;
  }

  .bet-line {
    color: #94a3b8;
    font-weight: 400;
  }

  .bet-odds-book {
    text-align: right;
  }

  .bet-odds-book .odds {
    display: block;
    font-size: 18px;
    color: #10b981;
    font-weight: 700;
  }

  .bet-odds-book .book {
    font-size: 12px;
    color: #64748b;
  }

  /* Line Comparison Section */
  .line-comparison-section {
    background: #0f172a;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 16px;
  }

  .comparison-header {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    font-weight: 600;
  }

  .comparison-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .comparison-row {
    display: flex;
    align-items: center;
  }

  .comparison-label {
    font-size: 12px;
    color: #94a3b8;
    width: 60px;
    font-weight: 500;
  }

  .comparison-values {
    display: flex;
    flex: 1;
    gap: 8px;
  }

  .comparison-item {
    flex: 1;
    text-align: center;
    padding: 6px 8px;
    border-radius: 6px;
    background: #1e293b;
  }

  .comparison-item .source {
    display: block;
    font-size: 9px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 2px;
  }

  .comparison-item .value {
    font-size: 13px;
    color: #e2e8f0;
    font-weight: 600;
  }

  .comparison-item.kenpom {
    border-left: 2px solid #3b82f6;
  }

  .comparison-item.vegas {
    border-left: 2px solid #f59e0b;
  }

  .comparison-item.diff {
    border-left: 2px solid #64748b;
    background: #1e293b;
  }

  .comparison-item.diff.significant {
    border-left-color: #10b981;
    background: #0f2920;
  }

  .comparison-item.diff.significant .value {
    color: #10b981;
  }

  /* Enhanced Metrics */
  .bet-metrics-enhanced {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .metric-card {
    background: #0f172a;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #334155;
  }

  .metric-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    margin-bottom: 6px;
  }

  .metric-icon {
    font-size: 14px;
  }

  .metric-name {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
  }

  .metric-value {
    display: block;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .edge-metric .metric-value { color: #10b981; }
  .ev-metric .metric-value { color: #3b82f6; }
  .kelly-metric .metric-value { color: #f59e0b; }

  .metric-explanation {
    font-size: 10px;
    color: #64748b;
    line-height: 1.3;
  }

  /* Value Explanation Section */
  .value-explanation {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 14px;
    border: 1px solid #3b82f680;
  }

  .explanation-header {
    font-size: 13px;
    color: #60a5fa;
    font-weight: 600;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .header-icon {
    font-size: 16px;
  }

  .explanation-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .explanation-list li {
    font-size: 13px;
    color: #94a3b8;
    padding: 6px 0 6px 20px;
    position: relative;
    line-height: 1.4;
  }

  .explanation-list li::before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #10b981;
    font-weight: bold;
  }

  /* Expand Stats Button */
  .expand-stats-btn {
    width: 100%;
    background: transparent;
    border: 1px dashed #334155;
    color: #64748b;
    padding: 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
    margin-bottom: 14px;
  }

  .expand-stats-btn:hover {
    background: #0f172a;
    color: #94a3b8;
    border-color: #64748b;
  }

  /* Expanded Stats */
  .expanded-stats {
    padding-top: 14px;
    border-top: 1px solid #334155;
    margin-bottom: 14px;
  }

  .stat-edges-summary {
    margin-bottom: 16px;
  }

  .edges-header {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 600;
  }

  .edges-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .edge-badge {
    padding: 8px 12px;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .edge-badge.home {
    background: #1e3a5f;
    border: 1px solid #3b82f680;
  }

  .edge-badge.away {
    background: #3f1f1f;
    border: 1px solid #ef444480;
  }

  .edge-badge.tempo {
    background: #78350f;
    border: 1px solid #f59e0b80;
  }

  .edge-category {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
  }

  .edge-team {
    font-size: 12px;
    font-weight: 600;
    color: #e2e8f0;
  }

  /* Major Stats Detailed */
  .major-stats-detailed {
    background: #0f172a;
    border-radius: 8px;
    padding: 14px;
  }

  .stats-header {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-weight: 600;
  }

  .stats-table {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stats-table-header {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 1fr;
    padding: 8px 4px;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    border-bottom: 1px solid #334155;
  }

  .stats-table-row {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 1fr;
    padding: 8px 4px;
    font-size: 12px;
    color: #94a3b8;
    border-radius: 4px;
  }

  .stats-table-row:hover {
    background: #1e293b;
  }

  .stats-table-row .stat-name {
    color: #e2e8f0;
    font-weight: 500;
  }

  .stats-table-row .advantage {
    color: #10b981;
    font-weight: 600;
  }

  .stats-table-row .stat-edge {
    font-weight: 600;
  }

  .stats-table-row.home .stat-edge {
    color: #60a5fa;
  }

  .stats-table-row.away .stat-edge {
    color: #f87171;
  }

  /* Confidence Section */
  .confidence-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }

  .confidence-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    padding: 8px 16px;
    border-radius: 20px;
    background: #334155;
    color: #94a3b8;
  }

  .confidence-icon {
    font-size: 14px;
  }

  .confidence-badge.high {
    background: linear-gradient(135deg, #065f46 0%, #047857 100%);
    color: #6ee7b7;
  }

  .confidence-badge.medium {
    background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
    color: #fcd34d;
  }

  .confidence-badge.low {
    background: #334155;
    color: #94a3b8;
  }

  .confidence-explanation {
    font-size: 11px;
    color: #64748b;
    text-align: center;
  }

  /* Tables */
  .games-table, .comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }

  .games-table th, .comparison-table th {
    text-align: left;
    padding: 12px;
    background: #1e293b;
    color: #94a3b8;
    font-weight: 500;
    border-bottom: 1px solid #334155;
  }

  .games-table td, .comparison-table td {
    padding: 12px;
    border-bottom: 1px solid #1e293b;
  }

  .games-table tr:hover, .comparison-table tr:hover {
    background: #1e293b;
  }

  .matchup {
    font-weight: 500;
  }

  .matchup .rank {
    color: #64748b;
    font-size: 12px;
  }

  .matchup .at {
    color: #64748b;
    margin: 0 8px;
  }

  .value-count {
    background: #10b981;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
  }

  .stat-diff-count {
    background: #f59e0b;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    cursor: help;
  }

  .no-values {
    color: #64748b;
  }

  .diff {
    color: #f59e0b;
    font-weight: 500;
  }

  .big-diff {
    color: #10b981;
    font-weight: 600;
  }

  .lean {
    font-weight: 500;
  }

  .lean-over {
    color: #10b981;
    font-weight: 600;
  }

  .lean-under {
    color: #ef4444;
    font-weight: 600;
  }

  /* Stat Differences View */
  .stat-diffs-view h2 {
    margin-bottom: 8px;
  }

  .stat-diffs-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .stat-diffs-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .stat-diff-game-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid #334155;
  }

  .stat-diff-game-card:hover {
    border-color: #3b82f6;
  }

  .stat-diff-game-card.selected {
    border-color: #3b82f6;
    background: #1e3a5f;
  }

  .stat-diff-game-card .game-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .stat-diff-game-card .game-matchup {
    font-weight: 600;
    font-size: 15px;
  }

  .stat-diff-game-card .major-count {
    background: #10b981;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
  }

  .major-diffs-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .diff-badge {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
  }

  .diff-badge.home {
    background: #1e40af;
    color: #93c5fd;
  }

  .diff-badge.away {
    background: #7f1d1d;
    color: #fca5a5;
  }

  .diff-badge.neutral {
    background: #334155;
    color: #94a3b8;
  }

  .more-badge {
    background: #334155;
    color: #94a3b8;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
  }

  /* Stat Comparison Detail */
  .stat-comparison-detail {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #334155;
  }

  .edge-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
  }

  .edge-item {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0f172a;
    padding: 8px 12px;
    border-radius: 8px;
  }

  .edge-item.tempo-mismatch {
    background: #78350f;
  }

  .edge-label {
    color: #94a3b8;
    font-size: 12px;
  }

  .edge-value {
    font-weight: 600;
    font-size: 13px;
  }

  .edge-value.home {
    color: #60a5fa;
  }

  .edge-value.away {
    color: #f87171;
  }

  /* Stats Detail Table */
  .stats-detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 12px;
  }

  .stats-detail-table th {
    text-align: left;
    padding: 10px 8px;
    background: #0f172a;
    color: #94a3b8;
    font-weight: 500;
    border-bottom: 1px solid #334155;
  }

  .stats-detail-table td {
    padding: 10px 8px;
    border-bottom: 1px solid #1e293b;
  }

  .stats-detail-table tr.major {
    background: rgba(16, 185, 129, 0.1);
  }

  .stats-detail-table tr.moderate {
    background: rgba(245, 158, 11, 0.05);
  }

  .stats-detail-table td.advantage {
    color: #10b981;
    font-weight: 600;
  }

  .stats-detail-table .diff-cell {
    font-weight: 500;
  }

  .stats-detail-table .diff-cell.major {
    color: #10b981;
  }

  .stats-detail-table .diff-cell.moderate {
    color: #f59e0b;
  }

  .stats-detail-table .edge-cell.home {
    color: #60a5fa;
    font-weight: 500;
  }

  .stats-detail-table .edge-cell.away {
    color: #f87171;
    font-weight: 500;
  }

  /* Stats Legend */
  .stats-legend {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin-top: 24px;
    border: 1px solid #334155;
  }

  .stats-legend h3 {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 12px;
    font-weight: 500;
  }

  .legend-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
  }

  .legend-section h4 {
    font-size: 13px;
    color: #e2e8f0;
    margin-bottom: 8px;
    font-weight: 600;
  }

  .stats-legend .legend-item {
    margin-bottom: 8px;
  }

  .stat-name {
    color: #10b981;
    font-weight: 600;
    font-size: 12px;
  }

  .stat-desc {
    color: #94a3b8;
    font-size: 12px;
    margin-left: 8px;
  }

  .footer {
    background: #1e293b;
    padding: 16px 0;
    text-align: center;
    color: #64748b;
    font-size: 12px;
    border-top: 1px solid #334155;
  }

  /* Confidence Score in Badge */
  .confidence-score {
    margin-left: 6px;
    font-size: 11px;
    opacity: 0.9;
  }

  /* Confidence Breakdown Toggle */
  .confidence-breakdown-toggle {
    background: transparent;
    border: 1px dashed #334155;
    color: #64748b;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 11px;
    margin-top: 10px;
    width: 100%;
    transition: all 0.2s;
  }

  .confidence-breakdown-toggle:hover {
    background: #0f172a;
    color: #94a3b8;
    border-color: #64748b;
  }

  /* Confidence Breakdown Component */
  .confidence-breakdown {
    margin-top: 12px;
    background: #0f172a;
    border-radius: 8px;
    padding: 14px;
    border: 1px solid #334155;
  }

  .breakdown-header {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .breakdown-bars {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .breakdown-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .breakdown-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .breakdown-label .label-text {
    font-size: 11px;
    color: #94a3b8;
  }

  .breakdown-label .label-score {
    font-size: 11px;
    font-weight: 600;
  }

  .breakdown-bar-container {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
  }

  .breakdown-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .breakdown-bar.penalty {
    opacity: 0.7;
  }

  .breakdown-detail {
    font-size: 10px;
    color: #64748b;
    font-style: italic;
  }

  .breakdown-detail.note {
    color: #94a3b8;
  }

  /* Public Betting Details */
  .public-betting-details {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #334155;
  }

  .public-header {
    font-size: 11px;
    color: #14b8a6;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 10px;
  }

  .public-stats {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  .public-stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .public-stat .stat-label {
    font-size: 10px;
    color: #64748b;
  }

  .public-stat .stat-value {
    font-size: 14px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .sharp-signal {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #1e3a5f;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    color: #60a5fa;
  }

  /* Track Bet Section */
  .track-bet-section {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #334155;
  }

  .track-bet-btn {
    width: 100%;
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border: 1px solid #3b82f6;
    color: #60a5fa;
    padding: 10px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.2s;
  }

  .track-bet-btn:hover {
    background: #3b82f6;
    color: white;
  }

  .tracked-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: #10b981;
    font-size: 13px;
    font-weight: 500;
    padding: 10px;
    background: #0f2920;
    border-radius: 8px;
    border: 1px solid #10b981;
  }

  .tracked-icon {
    font-size: 14px;
  }

  /* Performance Dashboard */
  .performance-dashboard {
    max-width: 900px;
    margin: 0 auto;
  }

  .dashboard-header {
    margin-bottom: 24px;
  }

  .dashboard-header .header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
  }

  .dashboard-header h2 {
    margin-bottom: 8px;
  }

  .dashboard-intro {
    color: #94a3b8;
    font-size: 14px;
  }

  .refresh-results-btn {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border: 1px solid #3b82f6;
    color: #60a5fa;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.2s;
  }

  .refresh-results-btn:hover:not(:disabled) {
    background: #3b82f6;
    color: white;
  }

  .refresh-results-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Stats Breakdown */
  .stats-breakdown {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .stats-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #334155;
  }

  .stats-card h4 {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 12px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .stat-rows {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .stat-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
    background: #0f172a;
    border-radius: 6px;
  }

  .stat-row .tier, .stat-row .type, .stat-row .period {
    flex: 1;
    font-size: 12px;
    color: #94a3b8;
  }

  .stat-row .tier.high { color: #6ee7b7; }
  .stat-row .tier.medium { color: #fcd34d; }
  .stat-row .tier.low { color: #94a3b8; }

  .stat-row .record {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    min-width: 60px;
    text-align: center;
  }

  .stat-row .rate {
    font-size: 13px;
    font-weight: 600;
    color: #10b981;
    min-width: 40px;
    text-align: right;
  }

  .stat-row.streak .streak-value {
    font-weight: 700;
    font-size: 14px;
  }

  .stat-row .streak-value.win { color: #10b981; }
  .stat-row .streak-value.loss { color: #ef4444; }

  /* Result Badge Small */
  .result-badge-small {
    font-size: 10px;
    font-weight: 700;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .result-badge-small.win {
    background: #065f46;
    color: #6ee7b7;
  }

  .result-badge-small.loss {
    background: #7f1d1d;
    color: #fca5a5;
  }

  .result-badge-small.push {
    background: #78350f;
    color: #fcd34d;
  }

  /* Score Display */
  .score-display {
    display: flex;
    gap: 16px;
    padding: 10px 12px;
    background: #0f172a;
    border-radius: 6px;
    margin-top: 12px;
    font-size: 13px;
  }

  .final-score {
    color: #e2e8f0;
    font-weight: 500;
  }

  .actual-margin, .actual-total {
    color: #94a3b8;
  }

  /* Pending Indicator */
  .pending-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px;
    background: #1e293b;
    border-radius: 6px;
    margin-top: 12px;
    color: #f59e0b;
    font-size: 13px;
  }

  .pending-icon {
    font-size: 14px;
  }

  .performance-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }

  .perf-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #334155;
  }

  .perf-card.total {
    border-color: #3b82f6;
    background: linear-gradient(135deg, #1e293b 0%, #1e3a5f 100%);
  }

  .perf-card.record {
    border-color: #10b981;
  }

  .perf-card.win-rate {
    border-color: #f59e0b;
  }

  .perf-value {
    font-size: 28px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
  }

  .perf-card.total .perf-value { color: #60a5fa; }
  .perf-card.record .perf-value { color: #10b981; }
  .perf-card.win-rate .perf-value { color: #f59e0b; }
  .perf-card.high-conf .perf-value { color: #6ee7b7; }
  .perf-card.avg-edge .perf-value { color: #8b5cf6; }

  .perf-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
  }

  /* CLV Summary Card */
  .clv-summary-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f2920 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    border: 1px solid #10b981;
  }

  .clv-summary-card h3 {
    font-size: 16px;
    color: #10b981;
    margin-bottom: 8px;
  }

  .clv-description {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 16px;
    line-height: 1.5;
  }

  .clv-stats {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }

  .clv-stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .clv-stat .clv-value {
    font-size: 24px;
    font-weight: 700;
    color: #10b981;
  }

  .clv-stat .clv-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
  }

  /* Performance Filters */
  .performance-filters {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
  }

  .performance-filters button {
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
  }

  .performance-filters button:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .performance-filters button.active {
    background: #3b82f6;
    border-color: #3b82f6;
    color: white;
  }

  /* Tracked Bets List */
  .tracked-bets-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .tracked-bet-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    border-left: 4px solid #64748b;
    transition: all 0.2s;
  }

  .tracked-bet-card.result-win {
    border-left-color: #10b981;
    background: linear-gradient(90deg, #0f2920 0%, #1e293b 30%);
  }

  .tracked-bet-card.result-loss {
    border-left-color: #ef4444;
    background: linear-gradient(90deg, #2a1010 0%, #1e293b 30%);
  }

  .tracked-bet-card.result-push {
    border-left-color: #f59e0b;
    background: linear-gradient(90deg, #2a2010 0%, #1e293b 30%);
  }

  .tracked-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
  }

  .tracked-matchup {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .tracked-teams {
    font-weight: 600;
    font-size: 14px;
    color: #e2e8f0;
  }

  .tracked-date {
    font-size: 12px;
    color: #64748b;
  }

  .remove-btn {
    background: transparent;
    border: none;
    color: #64748b;
    font-size: 20px;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }

  .remove-btn:hover {
    color: #ef4444;
  }

  .tracked-bet-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 12px;
  }

  .bet-detail {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .detail-type {
    font-size: 10px;
    color: #64748b;
    background: #0f172a;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: 600;
  }

  .detail-pick {
    font-size: 15px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .detail-odds {
    color: #10b981;
    font-weight: 600;
  }

  .bet-metrics-small {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }

  .bet-metrics-small .metric {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    background: #0f172a;
  }

  .bet-metrics-small .metric.edge { color: #10b981; }
  .bet-metrics-small .metric.ev { color: #3b82f6; }
  .bet-metrics-small .metric.conf-high { color: #6ee7b7; }
  .bet-metrics-small .metric.conf-medium { color: #fcd34d; }
  .bet-metrics-small .metric.conf-low { color: #94a3b8; }

  /* Result Section */
  .result-section {
    padding-top: 12px;
    border-top: 1px solid #334155;
  }

  .result-buttons {
    display: flex;
    gap: 8px;
  }

  .result-btn {
    flex: 1;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.2s;
    border: 1px solid;
  }

  .result-btn.win {
    background: transparent;
    border-color: #10b981;
    color: #10b981;
  }

  .result-btn.win:hover {
    background: #10b981;
    color: white;
  }

  .result-btn.loss {
    background: transparent;
    border-color: #ef4444;
    color: #ef4444;
  }

  .result-btn.loss:hover {
    background: #ef4444;
    color: white;
  }

  .result-btn.push {
    background: transparent;
    border-color: #f59e0b;
    color: #f59e0b;
  }

  .result-btn.push:hover {
    background: #f59e0b;
    color: white;
  }

  .result-badge {
    text-align: center;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 14px;
  }

  .result-badge.win {
    background: #065f46;
    color: #6ee7b7;
  }

  .result-badge.loss {
    background: #7f1d1d;
    color: #fca5a5;
  }

  .result-badge.push {
    background: #78350f;
    color: #fcd34d;
  }

  @media (max-width: 768px) {
    .tabs {
      flex-direction: column;
    }

    .filters {
      flex-direction: column;
      align-items: stretch;
    }

    .refresh-btn {
      margin-left: 0;
    }

    .games-table {
      font-size: 12px;
    }

    .games-table th, .games-table td {
      padding: 8px 4px;
    }

    .performance-summary {
      grid-template-columns: repeat(2, 1fr);
    }

    .tracked-bet-info {
      flex-direction: column;
      align-items: flex-start;
    }
  }

  .error-boundary {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
  }

  .error-boundary-content {
    background: #1e2a3a;
    border-radius: 12px;
    padding: 2rem;
    max-width: 500px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .error-boundary-content h2 {
    color: #ff6b6b;
    margin-bottom: 1rem;
  }

  .error-boundary-content p {
    color: #94a3b8;
    margin-bottom: 1.5rem;
  }

  .error-boundary-button {
    background: #3b82f6;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
  }

  .error-boundary-button:hover {
    background: #2563eb;
  }

  .error-details {
    margin-top: 1.5rem;
    text-align: left;
  }

  .error-details summary {
    cursor: pointer;
    color: #94a3b8;
    margin-bottom: 0.5rem;
  }

  .error-details pre {
    background: #0f1419;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.75rem;
    color: #ff6b6b;
  }
`;

// Wrap App with ErrorBoundary for graceful error handling
function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}

export default AppWithErrorBoundary;
