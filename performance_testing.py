import time
import statistics
import json
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime

from html_content_processor import HTMLParser, read_folder_path
from search_engine import NEREnhancedSearch, SearchEngine


class NetPerformanceTest:
    def __init__(self):
        self.initialization_times: List[float] = []
        self.search_times: Dict[str, List[float]] = {}
        self.index_build_times: List[float] = []
        self.memory_usage: List[float] = []
        self.results_log = []
        self.query = 'gta'
        self.folder_path = './videogames/'

    def measure_initialization_time(self) -> float:
        """Measure time taken to initialize the search engine"""
        start_time = time.time()

        # Initialize all components
        file_path_list = read_folder_path(self.folder_path)
        user_parser = HTMLParser(file_path_list)
        user_parser.parse_and_process_html(True)

        search_engine = NEREnhancedSearch(user_parser)
        search_engine.build_inverted_index()
        search_engine.debug_tf_idf()
        search_engine.write_entity_index_to_file()

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.initialization_times.append(elapsed_time)
        return elapsed_time

    def measure_search_time(self, search_engine: NEREnhancedSearch, query: str, num_iterations: int = 5) -> Dict:
        """Measure search performance for a given query"""
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            results = search_engine.search(query)
            end_time = time.time()
            times.append(end_time - start_time)

        self.search_times[query] = times

        return {
            'query': query,
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0
        }

    def run_full_performance_test(self) -> Dict:
        """Run a complete performance test suite"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'initialization': {},
            'searches': [],
            'index_stats': {}
        }

        # Test initialization
        init_time = self.measure_initialization_time()
        results['initialization'] = {
            'total_time': init_time,
            'folder_path': self.folder_path
        }

        # Initialize search engine for search tests
        file_path_list = read_folder_path(self.folder_path)
        user_parser = HTMLParser(file_path_list)
        user_parser.parse_and_process_html(True)
        search_engine = NEREnhancedSearch(user_parser)
        search_engine.build_inverted_index()

        # Test searches
        search_stats = self.measure_search_time(search_engine, self.query)
        results['searches'].append(search_stats)

        # Collect index statistics
        results['index_stats'] = {
            'total_documents': len(search_engine.docIDs),
            'vocabulary_size': len(search_engine.vocab),
            'unique_terms': len(search_engine.find_unique_terms()),
            'entity_types': len(search_engine.entity_index.keys())
        }

        self.results_log.append(results)
        return results

    def save_results(self, filename: str = "performance_results.json"):
        """Save performance test results to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results_log, f, indent=4)

    def plot_search_times(self, show_plot: bool = True, save_plot: bool = False):
        """Generate a box plot of search times for different queries"""
        plt.figure(figsize=(10, 6))
        plt.boxplot(self.search_times.values(), labels=self.search_times.keys())
        plt.title('Search Time Distribution by Query')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)

        if save_plot:
            plt.savefig('search_times_plot.png')
        if show_plot:
            plt.show()

    def generate_report(self) -> str:
        """Generate a formatted performance report"""
        report = []
        report.append("Search Engine Performance Report")
        report.append("=" * 30)

        # Initialization statistics
        report.append("\nInitialization Times:")
        report.append(f"Average: {statistics.mean(self.initialization_times):.4f} seconds")
        report.append(f"Min: {min(self.initialization_times):.4f} seconds")
        report.append(f"Max: {max(self.initialization_times):.4f} seconds")

        # Search performance
        report.append("\nSearch Performance:")
        for query, times in self.search_times.items():
            report.append(f"\nQuery: '{query}'")
            report.append(f"  Average time: {statistics.mean(times):.4f} seconds")
            report.append(f"  Min time: {min(times):.4f} seconds")
            report.append(f"  Max time: {max(times):.4f} seconds")
            if len(times) > 1:
                report.append(f"  Std dev: {statistics.stdev(times):.4f} seconds")

        return "\n".join(report)


    def default_tf_idf(self):
        file_path_list = read_folder_path(self.folder_path)
        user_parser = HTMLParser(file_path_list)
        user_parser.parse_and_process_html(True)

        search_engine = SearchEngine(user_parser)  # Use NEREnhancedSearch instead of SearchEngine
        search_engine.build_inverted_index(True)
        search_engine.debug_tf_idf()
        search_engine.user_prompt_tfidf()