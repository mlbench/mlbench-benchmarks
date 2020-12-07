# Changelog

## [v3.0.0](https://github.com/mlbench/mlbench-benchmarks/tree/v3.0.0) (2020-12-07)

[Full Changelog](https://github.com/mlbench/mlbench-benchmarks/compare/v2.0.0...v3.0.0)

**Implemented enhancements:**

- Update PyTorch base to 1.7 [\#64](https://github.com/mlbench/mlbench-benchmarks/issues/64)
- Add NLP/machine translation Transformer benchmark task [\#33](https://github.com/mlbench/mlbench-benchmarks/issues/33)
- Repair Logistic regression Model [\#30](https://github.com/mlbench/mlbench-benchmarks/issues/30)
- Add NLP/machine translation RNN benchmark task [\#27](https://github.com/mlbench/mlbench-benchmarks/issues/27)
- Add NLP benchmark images & task [\#24](https://github.com/mlbench/mlbench-benchmarks/issues/24)
- Add Gloo support to PyTorch images [\#23](https://github.com/mlbench/mlbench-benchmarks/issues/23)
- Add NCCL support to PyTorch images [\#22](https://github.com/mlbench/mlbench-benchmarks/issues/22)
- documentation: clearly link ref code to benchmark tasks [\#14](https://github.com/mlbench/mlbench-benchmarks/issues/14)
- Add time-to-accuracy speedup plot [\#7](https://github.com/mlbench/mlbench-benchmarks/issues/7)
- Update GKE documentation to use kubernetes version 1.10.9 [\#4](https://github.com/mlbench/mlbench-benchmarks/issues/4)
- Add tensorflow cifar10 benchmark [\#3](https://github.com/mlbench/mlbench-benchmarks/issues/3)
- Transformer language translation [\#51](https://github.com/mlbench/mlbench-benchmarks/pull/51) ([ehoelzl](https://github.com/ehoelzl))

**Fixed bugs:**

- Change Tensorflow Benchmark to use OpenMPI [\#8](https://github.com/mlbench/mlbench-benchmarks/issues/8)

**Closed issues:**

- Clean-up tasks [\#63](https://github.com/mlbench/mlbench-benchmarks/issues/63)
- Support for local run [\#59](https://github.com/mlbench/mlbench-benchmarks/issues/59)
- task implementations: delete choco, name tasks nlp/language-model  and nlp/translation [\#55](https://github.com/mlbench/mlbench-benchmarks/issues/55)
- remove open/closed division distinction [\#47](https://github.com/mlbench/mlbench-benchmarks/issues/47)
- \[Not an Issue\] Comparing 3 backends on multi-node single-gpu env [\#44](https://github.com/mlbench/mlbench-benchmarks/issues/44)
- Create light version of the base image for development [\#43](https://github.com/mlbench/mlbench-benchmarks/issues/43)
- No unit tests [\#40](https://github.com/mlbench/mlbench-benchmarks/issues/40)
- Remove stale branches [\#39](https://github.com/mlbench/mlbench-benchmarks/issues/39)
- Remove Communication backend from image name [\#36](https://github.com/mlbench/mlbench-benchmarks/issues/36)
- pytorch 1.4 [\#34](https://github.com/mlbench/mlbench-benchmarks/issues/34)
- create light version \(in addition to full\) for resource heavy benchmark tasks [\#19](https://github.com/mlbench/mlbench-benchmarks/issues/19)
- add script to compute official results from raw results \(time to acc for example\) [\#18](https://github.com/mlbench/mlbench-benchmarks/issues/18)

**Merged pull requests:**

- Add workflow [\#68](https://github.com/mlbench/mlbench-benchmarks/pull/68) ([ehoelzl](https://github.com/ehoelzl))
- Fix rnn language model [\#67](https://github.com/mlbench/mlbench-benchmarks/pull/67) ([ehoelzl](https://github.com/ehoelzl))
- Update pytorch [\#65](https://github.com/mlbench/mlbench-benchmarks/pull/65) ([ehoelzl](https://github.com/ehoelzl))
- Adapt optimizer imports [\#62](https://github.com/mlbench/mlbench-benchmarks/pull/62) ([ehoelzl](https://github.com/ehoelzl))
- Translation changes [\#61](https://github.com/mlbench/mlbench-benchmarks/pull/61) ([ehoelzl](https://github.com/ehoelzl))
- Change 'Benchmarks' to 'Benchmark Implementations' [\#60](https://github.com/mlbench/mlbench-benchmarks/pull/60) ([ehoelzl](https://github.com/ehoelzl))
- Add generic worker [\#58](https://github.com/mlbench/mlbench-benchmarks/pull/58) ([ehoelzl](https://github.com/ehoelzl))
- Rename tasks [\#57](https://github.com/mlbench/mlbench-benchmarks/pull/57) ([ehoelzl](https://github.com/ehoelzl))
- Add link to task description [\#56](https://github.com/mlbench/mlbench-benchmarks/pull/56) ([ehoelzl](https://github.com/ehoelzl))
- Fix tasks [\#54](https://github.com/mlbench/mlbench-benchmarks/pull/54) ([ehoelzl](https://github.com/ehoelzl))
- Add backend benchmark code and image [\#53](https://github.com/mlbench/mlbench-benchmarks/pull/53) ([ehoelzl](https://github.com/ehoelzl))
- Update nccl [\#52](https://github.com/mlbench/mlbench-benchmarks/pull/52) ([ehoelzl](https://github.com/ehoelzl))
- Remove open/closed division from benchmarks [\#49](https://github.com/mlbench/mlbench-benchmarks/pull/49) ([mmilenkoski](https://github.com/mmilenkoski))
- Pytorch 1.5.0 [\#48](https://github.com/mlbench/mlbench-benchmarks/pull/48) ([giorgiosav](https://github.com/giorgiosav))
- Refactor controlflow [\#46](https://github.com/mlbench/mlbench-benchmarks/pull/46) ([ehoelzl](https://github.com/ehoelzl))
- Add Image Recognition Benchmark with DistributedDataParallel [\#42](https://github.com/mlbench/mlbench-benchmarks/pull/42) ([mmilenkoski](https://github.com/mmilenkoski))
- Pytorch v1.4.0 [\#41](https://github.com/mlbench/mlbench-benchmarks/pull/41) ([ehoelzl](https://github.com/ehoelzl))
- Add aggregation by model [\#38](https://github.com/mlbench/mlbench-benchmarks/pull/38) ([ehoelzl](https://github.com/ehoelzl))
- Add NCCL & GLOO support to images [\#35](https://github.com/mlbench/mlbench-benchmarks/pull/35) ([giorgiosav](https://github.com/giorgiosav))
- Rnn language translation [\#32](https://github.com/mlbench/mlbench-benchmarks/pull/32) ([ehoelzl](https://github.com/ehoelzl))
- Linear model [\#28](https://github.com/mlbench/mlbench-benchmarks/pull/28) ([ehoelzl](https://github.com/ehoelzl))
- Fix ci [\#26](https://github.com/mlbench/mlbench-benchmarks/pull/26) ([ehoelzl](https://github.com/ehoelzl))
- \[WIP\]Add LSTM language model [\#25](https://github.com/mlbench/mlbench-benchmarks/pull/25) ([Panaetius](https://github.com/Panaetius))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
