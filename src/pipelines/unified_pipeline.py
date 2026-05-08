"""Unified pipeline orchestrating all three domain pipelines."""

import numpy as np
from typing import Dict, Any, Optional, List
import json
import time
from pathlib import Path

from .temporal_pipeline import TemporalPipeline
from .spectral_pipeline import SpectralPipeline
from .hybrid_pipeline import HybridPipeline
from ..evaluation.statistical_tests import mcnemar_test, compare_classifiers_comprehensive
from ..models.model_factory import ModelFactory
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class UnifiedPipeline:
    """
    Unified pipeline orchestrating temporal, spectral, and hybrid pipelines.
    
    Comprehensive Workflow:
    1. Run temporal-only pipeline
    2. Run spectral-only pipeline
    3. Run hybrid pipeline
    4. Statistical comparison (McNemar tests, effect sizes)
    5. Generate comparative analysis and conclusions
    
    Final Deliverable: Answer the question:
    "Which representation domain performs best, and why?"
    """
    
    def __init__(self, models: Optional[List[str]] = None,
                 output_dir: str = None):
        """
        Initialize unified pipeline.
        
        Args:
            models: List of model types to use. Default: ['random_forest', 'xgboost']
            output_dir: Directory to save results
        """
        self.models = models or ['random_forest', 'xgboost']
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipelines_results = {}
        self.comparison_results = {}
        self.best_results = {}
    
    def run_all_pipelines(self, X: np.ndarray, y: np.ndarray,
                         test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Run all three domain pipelines.
        
        Args:
            X: Input signals (n_samples, signal_length)
            y: Target labels
            test_size: Test set fraction
        
        Returns:
            Dictionary with results from all pipelines
        """
        logger.info("\n" + "=" * 70)
        logger.info("UNIFIED PIPELINE - ALL DOMAINS")
        logger.info("=" * 70 + "\n")
        
        results = {}
        
        # Run temporal pipeline
        for model_type in self.models:
            logger.info(f"\n>>> TEMPORAL DOMAIN with {model_type}")
            temporal = TemporalPipeline(model_type=model_type)
            temporal_results = temporal.run(X, y, test_size=test_size)
            results[f'temporal_{model_type}'] = temporal_results
        
        # Run spectral pipeline
        for model_type in self.models:
            logger.info(f"\n>>> SPECTRAL DOMAIN with {model_type}")
            spectral = SpectralPipeline(model_type=model_type)
            spectral_results = spectral.run(X, y, test_size=test_size)
            results[f'spectral_{model_type}'] = spectral_results
        
        # Run hybrid pipeline
        for model_type in self.models:
            logger.info(f"\n>>> HYBRID DOMAIN (Temporal + Spectral) with {model_type}")
            hybrid = HybridPipeline(model_type=model_type)
            hybrid_results = hybrid.run(X, y, test_size=test_size)
            results[f'hybrid_{model_type}'] = hybrid_results
        
        self.pipelines_results = results
        return results
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison across all pipelines and models.
        
        Returns:
            Comparison summary with rankings
        """
        logger.info("\n" + "=" * 70)
        logger.info("COMPARATIVE ANALYSIS")
        logger.info("=" * 70 + "\n")
        
        # Extract accuracies
        accuracies = {}
        for pipeline_name, results in self.pipelines_results.items():
            accuracy = results['overall']['accuracy']
            accuracies[pipeline_name] = accuracy
        
        # Rank pipelines
        sorted_pipelines = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
        
        summary = {
            'accuracies': accuracies,
            'ranking': [name for name, _ in sorted_pipelines],
            'ranking_scores': [score for _, score in sorted_pipelines],
            'best_pipeline': sorted_pipelines[0][0],
            'best_accuracy': sorted_pipelines[0][1],
        }
        
        # Analyze by domain
        summary['by_domain'] = self._analyze_by_domain()
        
        # Analyze by model
        summary['by_model'] = self._analyze_by_model()
        
        logger.info("\n=== RANKING ===")
        for i, (name, score) in enumerate(sorted_pipelines, 1):
            logger.info(f"{i}. {name}: {score:.3f}")
        
        self.comparison_results = summary
        return summary
    
    def _analyze_by_domain(self) -> Dict[str, Any]:
        """Analyze performance aggregated by domain."""
        domain_results = {'temporal': [], 'spectral': [], 'hybrid': []}
        
        for pipeline_name, results in self.pipelines_results.items():
            domain = results['pipeline']
            accuracy = results['overall']['accuracy']
            domain_results[domain].append(accuracy)
        
        analysis = {}
        for domain, accuracies in domain_results.items():
            if accuracies:
                analysis[domain] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'n_models': len(accuracies),
                }
        
        return analysis
    
    def _analyze_by_model(self) -> Dict[str, Any]:
        """Analyze performance aggregated by model type."""
        model_results = {model: [] for model in self.models}
        
        for pipeline_name, results in self.pipelines_results.items():
            # Extract model type from pipeline name (e.g., 'temporal_random_forest' -> 'random_forest')
            for model in self.models:
                if model in pipeline_name:
                    accuracy = results['overall']['accuracy']
                    model_results[model].append(accuracy)
                    break
        
        analysis = {}
        for model, accuracies in model_results.items():
            if accuracies:
                analysis[model] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'n_domains': len(accuracies),
                }
        
        return analysis
    
    def get_feature_analysis(self) -> Dict[str, int]:
        """Get feature counts for each domain."""
        feature_counts = {}
        
        for pipeline_name, results in self.pipelines_results.items():
            if 'n_features' in results:
                feature_counts[pipeline_name] = results['n_features']
        
        return feature_counts
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive text report.
        
        Args:
            output_file: Optional file to save report
        
        Returns:
            Report text
        """
        report = "\n" + "=" * 70 + "\n"
        report += "APPLIANCE POWER CLASSIFICATION - DOMAIN COMPARISON REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 70 + "\n"
        if self.comparison_results:
            best = self.comparison_results['best_pipeline']
            best_acc = self.comparison_results['best_accuracy']
            report += f"Best Pipeline: {best} ({best_acc:.3f})\n\n"
        
        # Domain Analysis
        report += "DOMAIN PERFORMANCE ANALYSIS\n"
        report += "-" * 70 + "\n"
        if self.comparison_results and 'by_domain' in self.comparison_results:
            for domain, stats in self.comparison_results['by_domain'].items():
                report += f"\n{domain.upper()}:\n"
                report += f"  Mean Accuracy: {stats['mean_accuracy']:.3f}\n"
                report += f"  Std Dev: {stats['std_accuracy']:.3f}\n"
                report += f"  Range: [{stats['min_accuracy']:.3f}, {stats['max_accuracy']:.3f}]\n"
        
        # Model Analysis
        report += "\n\nMODEL PERFORMANCE ANALYSIS\n"
        report += "-" * 70 + "\n"
        if self.comparison_results and 'by_model' in self.comparison_results:
            for model, stats in self.comparison_results['by_model'].items():
                report += f"\n{model.upper()}:\n"
                report += f"  Mean Accuracy: {stats['mean_accuracy']:.3f}\n"
                report += f"  Std Dev: {stats['std_accuracy']:.3f}\n"
                report += f"  Range: [{stats['min_accuracy']:.3f}, {stats['max_accuracy']:.3f}]\n"
        
        # Feature Analysis
        report += "\n\nFEATURE DIMENSIONALITY\n"
        report += "-" * 70 + "\n"
        feature_analysis = self.get_feature_analysis()
        for pipeline_name, n_features in sorted(feature_analysis.items()):
            report += f"{pipeline_name}: {n_features} features\n"
        
        # Ranking
        report += "\n\nFINAL RANKING\n"
        report += "-" * 70 + "\n"
        if self.comparison_results:
            for i, (name, score) in enumerate(
                zip(self.comparison_results['ranking'], 
                    self.comparison_results['ranking_scores']), 1):
                report += f"{i}. {name}: {score:.3f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        if output_file:
            Path(output_file).write_text(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def save_results(self, output_file: str = None) -> None:
        """
        Save all results to JSON file.
        
        Args:
            output_file: Path to save results
        """
        if not output_file:
            if not self.output_dir:
                logger.warning("No output directory specified")
                return
            output_file = self.output_dir / 'results.json'
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, results in self.pipelines_results.items():
            serializable_results[key] = self._make_json_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump({
                'pipeline_results': serializable_results,
                'comparison': self.comparison_results,
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, dict):
            return {k: UnifiedPipeline._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [UnifiedPipeline._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def run_complete_analysis(self, X: np.ndarray, y: np.ndarray,
                             test_size: float = 0.2,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            X: Input signals
            y: Target labels
            test_size: Test set fraction
            save_results: Whether to save results
        
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        # Run pipelines
        self.run_all_pipelines(X, y, test_size=test_size)
        
        # Generate comparison
        comparison = self.get_comparison_summary()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        if save_results and self.output_dir:
            self.save_results()
            report_file = self.output_dir / 'report.txt'
            Path(report_file).write_text(report)
        
        elapsed = time.time() - start_time
        logger.info(f"\nComplete analysis finished in {elapsed:.1f} seconds")
        
        return {
            'pipelines': self.pipelines_results,
            'comparison': self.comparison_results,
            'report': report,
            'elapsed_time': elapsed,
        }
