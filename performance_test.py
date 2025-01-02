from performance_testing import NetPerformanceTest

folder_path = './videogames/'

net_performance_test = NetPerformanceTest()
# elapsed_time = net_performance_test.measure_initialization_time()
# net_performance_test.run_full_performance_test()
# net_performance_test.plot_search_times()
# net_performance_test.save_results()

net_performance_test.default_tf_idf()

