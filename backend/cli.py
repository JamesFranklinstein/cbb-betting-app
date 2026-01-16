#!/usr/bin/env python3
"""
CBB Betting CLI

Command-line interface for getting daily picks and value bets.
"""

import asyncio
import sys
from datetime import date, datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from clients import KenPomClient
from services import GameService
from models import init_db

app = typer.Typer(help="College Basketball Betting Analysis CLI")
console = Console()


@app.command()
def today(
    min_edge: float = typer.Option(0.03, "--min-edge", "-e", help="Minimum edge to show"),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all games, not just value bets"),
):
    """Show today's games and value bets."""
    console.print("\n[bold blue]🏀 College Basketball Value Bets[/bold blue]")
    console.print(f"[dim]Date: {date.today().strftime('%A, %B %d, %Y')}[/dim]\n")
    
    async def run():
        try:
            service = GameService()
            
            with console.status("[bold green]Fetching data from KenPom and Odds API..."):
                analyses = await service.get_todays_analysis()
            
            if not analyses:
                console.print("[yellow]No games found for today.[/yellow]")
                return
            
            console.print(f"[green]Found {len(analyses)} games today[/green]\n")
            
            # Show value bets first
            value_count = 0
            for analysis in analyses:
                value_bets = [vb for vb in analysis.value_bets if vb.edge >= min_edge]
                if value_bets:
                    value_count += len(value_bets)
                    _print_game_with_values(analysis, value_bets)
            
            if value_count == 0:
                console.print("[yellow]No value bets found meeting minimum edge criteria.[/yellow]\n")
            else:
                console.print(f"\n[bold green]Total value bets found: {value_count}[/bold green]\n")
            
            # Show all games if requested
            if show_all:
                console.print("\n[bold]All Games Today:[/bold]")
                _print_games_table(analyses)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if "KENPOM_API_KEY" in str(e) or "ODDS_API_KEY" in str(e):
                console.print("\n[yellow]Make sure you have set your API keys in the .env file:[/yellow]")
                console.print("  KENPOM_API_KEY=your_key_here")
                console.print("  ODDS_API_KEY=your_key_here")
            raise
    
    asyncio.run(run())


@app.command()
def game(
    home: str = typer.Argument(..., help="Home team name"),
    away: str = typer.Argument(..., help="Away team name"),
    game_date: Optional[str] = typer.Option(None, "--date", "-d", help="Game date (YYYY-MM-DD)"),
):
    """Analyze a specific matchup."""
    async def run():
        try:
            service = GameService()
            
            with console.status(f"[bold green]Analyzing {away} @ {home}..."):
                analysis = await service.get_game_analysis(home, away, game_date)
            
            if not analysis:
                console.print(f"[red]Could not find game: {away} @ {home}[/red]")
                return
            
            console.print(service.format_analysis(analysis))
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run())


@app.command()
def spreads(
    min_diff: float = typer.Option(2.0, "--min-diff", "-d", help="Minimum spread difference"),
):
    """Find games where KenPom spread differs significantly from Vegas."""
    async def run():
        try:
            service = GameService()
            
            with console.status("[bold green]Fetching spread comparisons..."):
                analyses = await service.get_todays_analysis()
            
            # Filter to games with significant spread differences
            spread_games = [
                a for a in analyses 
                if a.vegas_spread != 0 and abs(a.spread_diff) >= min_diff
            ]
            
            if not spread_games:
                console.print(f"[yellow]No games found with spread difference >= {min_diff}[/yellow]")
                return
            
            # Sort by spread difference
            spread_games.sort(key=lambda x: abs(x.spread_diff), reverse=True)
            
            table = Table(title="Spread Discrepancies", box=box.ROUNDED)
            table.add_column("Game", style="cyan")
            table.add_column("KenPom", justify="center")
            table.add_column("Vegas", justify="center")
            table.add_column("Diff", justify="center")
            table.add_column("Lean", justify="center")
            
            for a in spread_games:
                diff_color = "green" if abs(a.spread_diff) >= 3 else "yellow"
                
                # Determine lean
                if a.spread_diff > 0:
                    lean = f"{a.away_team} +{abs(a.vegas_spread):.1f}"
                else:
                    lean = f"{a.home_team} {a.vegas_spread:.1f}"
                
                table.add_row(
                    f"{a.away_team} @ {a.home_team}",
                    f"{a.home_team} {-a.kenpom_spread:+.1f}",
                    f"{a.home_team} {a.vegas_spread:+.1f}",
                    f"[{diff_color}]{a.spread_diff:+.1f}[/{diff_color}]",
                    lean
                )
            
            console.print(table)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run())


@app.command()
def totals(
    min_diff: float = typer.Option(3.0, "--min-diff", "-d", help="Minimum total difference"),
):
    """Find games where KenPom total differs significantly from Vegas."""
    async def run():
        try:
            service = GameService()
            
            with console.status("[bold green]Fetching total comparisons..."):
                analyses = await service.get_todays_analysis()
            
            # Filter to games with significant total differences
            total_games = [
                a for a in analyses 
                if a.vegas_total != 0 and abs(a.total_diff) >= min_diff
            ]
            
            if not total_games:
                console.print(f"[yellow]No games found with total difference >= {min_diff}[/yellow]")
                return
            
            # Sort by total difference
            total_games.sort(key=lambda x: abs(x.total_diff), reverse=True)
            
            table = Table(title="Total Discrepancies", box=box.ROUNDED)
            table.add_column("Game", style="cyan")
            table.add_column("KenPom", justify="center")
            table.add_column("Vegas", justify="center")
            table.add_column("Diff", justify="center")
            table.add_column("Lean", justify="center")
            
            for a in total_games:
                diff_color = "green" if abs(a.total_diff) >= 5 else "yellow"
                lean = "OVER" if a.total_diff > 0 else "UNDER"
                lean_color = "green" if lean == "OVER" else "red"
                
                table.add_row(
                    f"{a.away_team} @ {a.home_team}",
                    f"{a.kenpom_total:.1f}",
                    f"{a.vegas_total:.1f}",
                    f"[{diff_color}]{a.total_diff:+.1f}[/{diff_color}]",
                    f"[{lean_color}]{lean}[/{lean_color}]"
                )
            
            console.print(table)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run())


def _get_current_season_year() -> int:
    """
    Get the current college basketball season year.

    The season year is the year the season ends (e.g., 2025-26 season = 2026).
    If before July, we're in the season ending this year.
    If July or later, we're in the season ending next year.
    """
    today = date.today()
    return today.year if today.month < 7 else today.year + 1


@app.command()
def rankings(
    conference: Optional[str] = typer.Option(None, "--conf", "-c", help="Filter by conference"),
    top_n: int = typer.Option(25, "--top", "-n", help="Number of teams to show"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Season year (defaults to current)"),
):
    """Show current KenPom rankings."""
    async def run():
        try:
            kenpom = KenPomClient()

            # Use provided year or detect current season
            season_year = year if year else _get_current_season_year()
            params = {"year": season_year}
            if conference:
                params["conference"] = conference

            with console.status("[bold green]Fetching rankings..."):
                ratings = await kenpom.get_ratings(**params)
            
            # Sort by rank
            ratings.sort(key=lambda x: x.get("RankAdjEM", 999))
            ratings = ratings[:top_n]

            table = Table(title=f"KenPom Rankings {season_year-1}-{str(season_year)[2:]} (Top {top_n})", box=box.ROUNDED)
            table.add_column("Rank", justify="right", style="cyan")
            table.add_column("Team", style="white")
            table.add_column("Conf", justify="center")
            table.add_column("Record", justify="center")
            table.add_column("AdjEM", justify="right")
            table.add_column("AdjO", justify="right")
            table.add_column("AdjD", justify="right")
            table.add_column("Tempo", justify="right")
            
            for team in ratings:
                table.add_row(
                    str(team.get("RankAdjEM", "")),
                    team.get("TeamName", ""),
                    team.get("ConfShort", ""),
                    f"{team.get('Wins', 0)}-{team.get('Losses', 0)}",
                    f"{team.get('AdjEM', 0):+.2f}",
                    f"{team.get('AdjOE', 0):.1f}",
                    f"{team.get('AdjDE', 0):.1f}",
                    f"{team.get('AdjTempo', 0):.1f}",
                )
            
            console.print(table)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run())


@app.command()
def init():
    """Initialize the database."""
    try:
        init_db()
        console.print("[green]Database initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error initializing database: {e}[/red]")


# ==================== HELPER FUNCTIONS ====================

def _print_game_with_values(analysis, value_bets):
    """Print a game with its value bets."""
    # Game header
    time_str = analysis.game_time.strftime("%I:%M %p")
    header = f"#{analysis.away_rank} {analysis.away_team} @ #{analysis.home_rank} {analysis.home_team} ({time_str})"
    
    console.print(Panel(header, style="bold cyan", box=box.ROUNDED))
    
    # KenPom vs Vegas comparison
    console.print(f"  [dim]KenPom:[/dim] {analysis.home_team} {-analysis.kenpom_spread:+.1f} | Total: {analysis.kenpom_total:.1f} | Win: {analysis.kenpom_home_win_prob:.0%}")
    if analysis.vegas_spread != 0:
        console.print(f"  [dim]Vegas:[/dim]  {analysis.home_team} {analysis.vegas_spread:+.1f} | Total: {analysis.vegas_total:.1f} | ML: {analysis.vegas_ml_home:+d}/{analysis.vegas_ml_away:+d}")
    
    # Value bets
    console.print()
    for vb in value_bets:
        _print_value_bet(vb)
    console.print()


def _print_value_bet(vb):
    """Print a single value bet."""
    confidence_colors = {"high": "green", "medium": "yellow", "low": "white"}
    conf_color = confidence_colors.get(vb.confidence, "white")
    
    if vb.bet_type.value == "spread":
        line_str = f"{vb.market_line:+.1f}" if vb.market_line else ""
        bet_str = f"📊 SPREAD: {vb.side} {line_str}"
    elif vb.bet_type.value == "total":
        bet_str = f"📈 TOTAL: {vb.side.upper()} {vb.market_line}"
    else:
        bet_str = f"💰 ML: {vb.side}"
    
    console.print(f"  [{conf_color}]{bet_str}[/{conf_color}]")
    console.print(f"    Model: {vb.model_prob:.1%} | Market: {vb.market_implied_prob:.1%} | [bold]Edge: {vb.edge:.1%}[/bold]")
    console.print(f"    Odds: {vb.market_odds:+d} @ {vb.recommended_book} | EV: {vb.ev:+.1%} | Kelly: {vb.kelly_fraction:.1%}")


def _print_games_table(analyses):
    """Print a table of all games."""
    table = Table(box=box.SIMPLE)
    table.add_column("Time", style="dim")
    table.add_column("Game", style="cyan")
    table.add_column("KenPom Spread", justify="center")
    table.add_column("Vegas Spread", justify="center")
    table.add_column("Total", justify="center")
    table.add_column("Values", justify="center")
    
    for a in analyses:
        time_str = a.game_time.strftime("%I:%M %p")
        game_str = f"#{a.away_rank} {a.away_team} @ #{a.home_rank} {a.home_team}"
        
        vegas_spread = f"{a.vegas_spread:+.1f}" if a.vegas_spread != 0 else "-"
        vegas_total = f"{a.vegas_total:.1f}" if a.vegas_total != 0 else "-"
        
        value_count = len(a.value_bets)
        value_str = f"[green]{value_count}[/green]" if value_count > 0 else "[dim]0[/dim]"
        
        table.add_row(
            time_str,
            game_str,
            f"{a.home_team} {-a.kenpom_spread:+.1f}",
            f"{a.home_team} {vegas_spread}",
            f"KP: {a.kenpom_total:.1f} / V: {vegas_total}",
            value_str
        )
    
    console.print(table)


if __name__ == "__main__":
    app()
