"""
Output utilities for saving and displaying analysis results.

This module handles exporting analysis results to various formats
including CSV, JSON, and formatted console reports.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from .models import SessionAnalysisResult, ZoneAnalysisResult
from .logging_config import get_logger, log_function_call, console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

logger = get_logger("output")


class ResultsExporter:
    """
    Handles exporting and displaying analysis results.
    """
    
    def __init__(self):
        """Initialize the results exporter."""
        self.logger = get_logger("output")
    
    @log_function_call(logger)
    def export_to_csv(
        self, 
        results: List[SessionAnalysisResult], 
        output_file: Path
    ) -> None:
        """
        Export analysis results to CSV format.
        
        Args:
            results: List of session analysis results
            output_file: Path to output CSV file
        """
        self.logger.info(f"Exporting {len(results)} results to CSV: {output_file}")
        
        # Prepare data for CSV export
        rows = []
        for result in results:
            base_row = {
                'session_name': result.session_name,
                'total_duration_sec': result.total_duration,
                'total_distance_mm': result.total_distance,
                'mean_speed_mm_per_sec': result.mean_speed
            }
            
            # Add zone data
            for zone_result in result.zone_results:
                row = base_row.copy()
                row.update({
                    'zone_name': zone_result.zone_name,
                    'total_time_inside_sec': zone_result.total_time_inside,
                    'total_time_outside_sec': zone_result.total_time_outside,
                    'percentage_inside': zone_result.percentage_inside,
                    'entry_count': zone_result.entry_count,
                    'exit_count': zone_result.exit_count,
                    'mean_visit_duration_sec': zone_result.mean_visit_duration,
                    'longest_visit_sec': zone_result.longest_visit,
                    'shortest_visit_sec': zone_result.shortest_visit,
                    'distance_inside_mm': zone_result.distance_inside,
                    'distance_outside_mm': zone_result.distance_outside,
                    'mean_speed_inside_mm_per_sec': zone_result.mean_speed_inside,
                    'mean_speed_outside_mm_per_sec': zone_result.mean_speed_outside
                })
                rows.append(row)
        
        # Write to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Successfully exported data to {output_file}")
        else:
            self.logger.warning("No data to export")
    
    @log_function_call(logger)
    def export_to_json(
        self, 
        results: List[SessionAnalysisResult], 
        output_file: Path,
        include_summary: bool = True
    ) -> None:
        """
        Export analysis results to JSON format.
        
        Args:
            results: List of session analysis results
            output_file: Path to output JSON file
            include_summary: Whether to include summary statistics
        """
        self.logger.info(f"Exporting {len(results)} results to JSON: {output_file}")
        
        # Convert results to dictionaries
        export_data = {
            'sessions': [result.to_dict() for result in results],
            'metadata': {
                'total_sessions': len(results),
                'export_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Add summary statistics if requested
        if include_summary and results:
            from .analyzer import ToxTracAnalyzer
            analyzer = ToxTracAnalyzer()
            summary = analyzer.create_summary_statistics(results)
            export_data['summary'] = summary
        
        # Write to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Successfully exported data to {output_file}")
    
    @log_function_call(logger)
    def display_session_summary(self, result: SessionAnalysisResult) -> None:
        """
        Display a formatted summary of a single session analysis.
        
        Args:
            result: Session analysis result to display
        """
        # Create main info panel
        info_text = f"""
[bold]Session:[/bold] {result.session_name}
[bold]Duration:[/bold] {result.total_duration:.2f} seconds
"""
        
        if result.total_distance is not None:
            info_text += f"[bold]Total Distance:[/bold] {result.total_distance:.2f} mm\n"
        
        if result.mean_speed is not None:
            info_text += f"[bold]Mean Speed:[/bold] {result.mean_speed:.2f} mm/s\n"
        
        info_panel = Panel(info_text.strip(), title="Session Information", border_style="blue")
        
        # Create zone results table
        table = Table(title="Zone Analysis Results")
        table.add_column("Zone", style="cyan", no_wrap=True)
        table.add_column("Time Inside (s)", justify="right")
        table.add_column("Percentage (%)", justify="right")
        table.add_column("Entries", justify="right")
        table.add_column("Avg Visit (s)", justify="right")
        table.add_column("Max Visit (s)", justify="right")
        
        for zone_result in result.zone_results:
            table.add_row(
                zone_result.zone_name,
                f"{zone_result.total_time_inside:.2f}",
                f"{zone_result.percentage_inside:.1f}",
                str(zone_result.entry_count),
                f"{zone_result.mean_visit_duration:.2f}",
                f"{zone_result.longest_visit:.2f}"
            )
        
        # Display both panels
        console.print(info_panel)
        console.print(table)
        console.print()
    
    @log_function_call(logger)
    def display_summary_report(
        self, 
        results: List[SessionAnalysisResult],
        summary_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Display a comprehensive summary report for multiple sessions.
        
        Args:
            results: List of session analysis results
            summary_stats: Optional pre-calculated summary statistics
        """
        console.print(Panel.fit("📊 ToxTrac Analysis Summary Report", style="bold magenta"))
        
        # Calculate summary stats if not provided
        if summary_stats is None and results:
            from .analyzer import ToxTracAnalyzer
            analyzer = ToxTracAnalyzer()
            summary_stats = analyzer.create_summary_statistics(results)
        
        if not summary_stats:
            console.print("❌ No data to display", style="red")
            return
        
        # Overall statistics
        session_stats = summary_stats.get('session_statistics', {})
        
        overall_info = f"""
[bold]Total Sessions:[/bold] {summary_stats.get('n_sessions', 0)}
[bold]Mean Duration:[/bold] {session_stats.get('mean_duration', 0):.2f} ± {session_stats.get('std_duration', 0):.2f} seconds
"""
        
        if session_stats.get('mean_distance') is not None:
            overall_info += f"[bold]Mean Distance:[/bold] {session_stats.get('mean_distance', 0):.2f} ± {session_stats.get('std_distance', 0):.2f} mm\n"
        
        if session_stats.get('mean_speed') is not None:
            overall_info += f"[bold]Mean Speed:[/bold] {session_stats.get('mean_speed', 0):.2f} ± {session_stats.get('std_speed', 0):.2f} mm/s\n"
        
        console.print(Panel(overall_info.strip(), title="Overall Statistics", border_style="green"))
        
        # Zone statistics table
        zone_stats = summary_stats.get('zone_statistics', {})
        
        if zone_stats:
            table = Table(title="Zone Statistics Summary")
            table.add_column("Zone", style="cyan")
            table.add_column("N Sessions", justify="right")
            table.add_column("Mean % Inside", justify="right")
            table.add_column("Std % Inside", justify="right")
            table.add_column("Mean Entries", justify="right")
            table.add_column("Mean Visit Duration", justify="right")
            
            for zone_name, stats in zone_stats.items():
                table.add_row(
                    zone_name,
                    str(stats['n_sessions']),
                    f"{stats['mean_percentage_inside']:.1f}",
                    f"{stats['std_percentage_inside']:.1f}",
                    f"{stats['mean_entry_count']:.1f}",
                    f"{stats['mean_visit_duration']:.2f}"
                )
            
            console.print(table)
        
        console.print()
    
    @log_function_call(logger)
    def display_individual_sessions(self, results: List[SessionAnalysisResult]) -> None:
        """
        Display detailed results for each individual session.
        
        Args:
            results: List of session analysis results
        """
        console.print(Panel.fit("📋 Individual Session Results", style="bold blue"))
        
        for i, result in enumerate(results, 1):
            console.print(f"\n[bold cyan]Session {i}/{len(results)}[/bold cyan]")
            self.display_session_summary(result)
    
    @log_function_call(logger)
    def save_report_to_directory(
        self,
        results: List[SessionAnalysisResult],
        output_dir: Path,
        formats: List[str] = None
    ) -> List[Path]:
        """
        Save analysis results to a specific directory.
        
        Args:
            results: List of session analysis results
            output_dir: Directory to save reports (should already exist)
            formats: List of formats to save ('csv', 'json'). Defaults to both.
            
        Returns:
            List of paths to saved files
        """
        if formats is None:
            formats = ['csv', 'json']
        
        saved_files = []
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in requested formats
        if 'csv' in formats:
            csv_file = output_dir / "analysis_results.csv"
            self.export_to_csv(results, csv_file)
            saved_files.append(csv_file)
        
        if 'json' in formats:
            json_file = output_dir / "analysis_results.json"
            self.export_to_json(results, json_file, include_summary=True)
            saved_files.append(json_file)
        
        self.logger.info(f"Saved {len(saved_files)} report files to {output_dir}")
        return saved_files

    @log_function_call(logger)
    def save_report(
        self,
        results: List[SessionAnalysisResult],
        output_dir: Path,
        formats: List[str] = None
    ) -> List[Path]:
        """
        Save analysis results in multiple formats.
        
        Args:
            results: List of session analysis results
            output_dir: Directory to save reports
            formats: List of formats to save ('csv', 'json'). Defaults to both.
            
        Returns:
            List of paths to saved files
        """
        if formats is None:
            formats = ['csv', 'json']
        
        saved_files = []
        
        # Create timestamped output directory
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_dir = output_dir / f"toxtrac_analysis_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in requested formats
        if 'csv' in formats:
            csv_file = report_dir / "analysis_results.csv"
            self.export_to_csv(results, csv_file)
            saved_files.append(csv_file)
        
        if 'json' in formats:
            json_file = report_dir / "analysis_results.json"
            self.export_to_json(results, json_file, include_summary=True)
            saved_files.append(json_file)
        
        self.logger.info(f"Saved {len(saved_files)} report files to {report_dir}")
        return saved_files
