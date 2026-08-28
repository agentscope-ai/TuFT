切换方式总表
Method	T→S hot switch	S→T hot switch	Round-trip hot switch	Notes
Static coexistence	0.0 ms	0.0 ms	0.0 ms	无切换，但显存打满，training OOM
Process reload	219749.8 ms	80296.8 ms	300046.6 ms	kill + reload
vLLM sleep/wake + FSDP simulated reload	6091.7 ms	5325.0 ms	11416.7 ms	vLLM sleep/wake 真实，FSDP 模拟
FlexGPU zero-copy	370.3 ms	205.5 ms	575.8 ms	真实双向 zero-copy
FlexGPU hot breakdown
Direction	Release/build	Descriptor	Alias/inject	Total
T→S	308.1 ms	3.9 ms	58.3 ms	370.3 ms
S→T	201.0 ms	0.0 ms	4.5 ms	205.5 ms
Static coexistence memory pressure
Config	Training resident	Sampling budget	Total pressure	Result
32B TP2/DP2	32.0 GB	48.0 GB	80.0 GB	Training OOM
32B TP4/DP1	17.5 GB	62.5 GB	80.0 GB	Training OOM
