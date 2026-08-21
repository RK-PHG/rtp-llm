py_test(
    name = "test_mimo_v25",
    srcs = ["test_mimo_v25.py"],
    timeout = "eternal",
    data = [
        "//:th_transformer",
        "//:rtp_compute_ops",
    ],
    deps = [
        "//rtp_llm:rtp_llm_lib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:test_util",
    ],
)
