from __future__ import annotations
import argparse, json
from pathlib import Path
from .analysis import check_reproducibility, filter_trajectory, load_jsonl, summarize_trajectory, verify_replay
from .benchmark import GeneralizationPlan, SeedPartition, audit_generalization_report, benchmark_parallel_scaling, benchmark_remote, compare_benchmarks, compare_generalization_reports, evaluate_generalization_remote, generated_hazard_kinds, generated_mechanics_signature, generated_room_count, generated_world_validation, generator_config_fingerprint, summarize_generalization
from .datasets import export_csv, export_jsonl, export_parquet, summarize_world_model_dataset
from .experiments import ExperimentManifest, audit_experiment_manifest, reconcile_manifest_provenance
from .environment import EmbodiedEnv, EmbodiedEnvConfig
from .learning import TabularQConfig, TabularQPolicy, checkpoint_fingerprint, evaluate_tabular_partitions, evaluate_tabular_q, train_tabular_q
from .paths import ROOT
from .providers import CautiousProvider, ExplorerProvider, GeminiProvider, ScriptedProvider, RandomValidProvider, MockReasoningProvider
from .runner import run_remote
from .schemas import RewardConfig

def provider_for(name: str, seed: int, model: str | None = None):
    providers = {
        "scripted": ScriptedProvider,
        "mock_reasoning": MockReasoningProvider,
        "random_valid": lambda: RandomValidProvider(seed),
        "cautious": CautiousProvider,
        "explorer": ExplorerProvider,
        "gemini": lambda: GeminiProvider(model=model),
    }
    try:
        return providers[name]()
    except KeyError as error:
        raise ValueError(f"Unknown provider: {name}") from error

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    provider_choices=['scripted','mock_reasoning','random_valid','cautious','explorer','gemini']
    a=sub.add_parser('run'); source=a.add_mutually_exclusive_group(); source.add_argument('--scenario',default=None); source.add_argument('--generated-world-seed',type=int); a.add_argument('--generator-config',type=Path); a.add_argument('--reward-config',type=Path,help='JSON file containing a complete authoritative reward profile.'); a.add_argument('--seed',type=int,default=42); a.add_argument('--provider',choices=provider_choices,default='scripted'); a.add_argument('--model'); a.add_argument('--max-steps',type=int); a.add_argument('--max-wall-seconds',type=float); a.add_argument('--max-total-tokens',type=int); resume=a.add_mutually_exclusive_group(); resume.add_argument('--resume-run-id'); resume.add_argument('--restore-replay-id'); a.add_argument('--observation-mode',choices=['minimal','normal','rich','oracle','noisy'],default='normal'); a.add_argument('--include-research-snapshots',action='store_true',help='Export privileged before/after snapshots for offline world-model research. Never sends them to the provider.'); a.add_argument('--output',type=Path); a.add_argument('--memory-mode',choices=['none','recent'],default='recent'); a.add_argument('--server-url')
    a.add_argument('--memory-window',type=int,default=5,choices=range(1,101),metavar='1..100'); a.add_argument('--policy-state-mode',choices=['reset','preserve'],default='reset',help='Reset policy-local state before this run, or preserve it for an explicitly labeled continual-memory experiment.')
    b=sub.add_parser('benchmark'); b.add_argument('--scenario',default='survival_room'); b.add_argument('--provider',choices=provider_choices,default='scripted'); b.add_argument('--runs',type=int,default=5); b.add_argument('--seed-start',type=int,default=1000); b.add_argument('--concurrency',type=int,default=1); b.add_argument('--output',type=Path,default=ROOT/'data'/'exports'); b.add_argument('--server-url')
    scale=sub.add_parser('benchmark-scale'); scale.add_argument('--provider',choices=provider_choices,default='scripted'); scale.add_argument('--runs',type=int,default=5); scale.add_argument('--seed-start',type=int,default=1000); scale.add_argument('--workers',default='1,8,32,64'); scale.add_argument('--output',type=Path,default=ROOT/'data'/'exports'/'parallel-scaling'); scale.add_argument('--server-url')
    generalize=sub.add_parser('generalize'); generalize.add_argument('--provider',choices=provider_choices,default='scripted'); generalize.add_argument('--model',help='Provider model identifier, used by Gemini and persisted in the report.'); generalize.add_argument('--train-start',type=int,default=0); generalize.add_argument('--train-count',type=int,default=5); generalize.add_argument('--validation-start',type=int,default=8_000); generalize.add_argument('--validation-count',type=int,default=2); generalize.add_argument('--test-start',type=int,default=9_000); generalize.add_argument('--test-count',type=int,default=2); generalize.add_argument('--concurrency',type=int,default=1); generalize.add_argument('--observation-mode',choices=['minimal','normal','rich','oracle','noisy'],default='normal'); generalize.add_argument('--generator-config',type=Path); generalize.add_argument('--train-generator-config',type=Path); generalize.add_argument('--validation-generator-config',type=Path); generalize.add_argument('--test-generator-config',type=Path); generalize.add_argument('--output',type=Path,default=ROOT/'data'/'exports'/'generalization'); generalize.add_argument('--server-url')
    train_q=sub.add_parser('train-tabular'); train_q.add_argument('--seed-start',type=int,default=0); train_q.add_argument('--episodes',type=int,default=20); train_q.add_argument('--max-steps',type=int,default=128); train_q.add_argument('--learning-rate',type=float,default=.2); train_q.add_argument('--discount',type=float,default=.95); train_q.add_argument('--epsilon',type=float,default=.2); train_q.add_argument('--observation-mode',choices=['minimal','normal','rich','oracle','noisy'],default='normal'); train_q.add_argument('--generator-config',type=Path); train_q.add_argument('--checkpoint',type=Path,default=ROOT/'data'/'checkpoints'/'tabular_q.json'); train_q.add_argument('--server-url')
    eval_q=sub.add_parser('evaluate-tabular'); eval_q.add_argument('--checkpoint',type=Path,required=True); eval_q.add_argument('--seed-start',type=int,default=9000); eval_q.add_argument('--episodes',type=int,default=10); eval_q.add_argument('--max-steps',type=int,default=128); eval_q.add_argument('--observation-mode',choices=['minimal','normal','rich','oracle','noisy'],default='normal'); eval_q.add_argument('--generator-config',type=Path); eval_q.add_argument('--server-url')
    tabular_generalize=sub.add_parser('generalize-tabular'); tabular_generalize.add_argument('--train-start',type=int,default=0); tabular_generalize.add_argument('--train-count',type=int,default=20); tabular_generalize.add_argument('--validation-start',type=int,default=8_000); tabular_generalize.add_argument('--validation-count',type=int,default=10); tabular_generalize.add_argument('--test-start',type=int,default=9_000); tabular_generalize.add_argument('--test-count',type=int,default=10); tabular_generalize.add_argument('--max-steps',type=int,default=128); tabular_generalize.add_argument('--learning-rate',type=float,default=.2); tabular_generalize.add_argument('--discount',type=float,default=.95); tabular_generalize.add_argument('--epsilon',type=float,default=.2); tabular_generalize.add_argument('--observation-mode',choices=['minimal','normal','rich','oracle','noisy'],default='normal'); tabular_generalize.add_argument('--generator-config',type=Path); tabular_generalize.add_argument('--checkpoint',type=Path,default=ROOT/'data'/'checkpoints'/'tabular_q_generalization.json'); tabular_generalize.add_argument('--output',type=Path,default=ROOT/'data'/'exports'/'tabular-generalization'); tabular_generalize.add_argument('--server-url')
    c=sub.add_parser('analyze'); c.add_argument('--trajectory',type=Path,required=True); c.add_argument('--output',type=Path)
    d=sub.add_parser('compare'); d.add_argument('--left',type=Path,required=True); d.add_argument('--right',type=Path,required=True); d.add_argument('--output',type=Path)
    generalization_compare=sub.add_parser('compare-generalization'); generalization_compare.add_argument('--left',type=Path,required=True); generalization_compare.add_argument('--right',type=Path,required=True); generalization_compare.add_argument('--output',type=Path)
    generalization_audit=sub.add_parser('audit-generalization'); generalization_audit.add_argument('--report',type=Path,required=True); generalization_audit.add_argument('--checkpoint',type=Path,help='Verify a tabular checkpoint against checkpoint_fingerprint in the report.'); generalization_audit.add_argument('--output',type=Path)
    e=sub.add_parser('filter'); e.add_argument('--trajectory',type=Path,required=True); e.add_argument('--output',type=Path,required=True); e.add_argument('--action-type'); e.add_argument('--event-type'); e.add_argument('--valid-only',action='store_true'); e.add_argument('--csv',action='store_true')
    dataset_audit=sub.add_parser('audit-dataset'); dataset_audit.add_argument('--trajectory',type=Path,required=True); dataset_audit.add_argument('--output',type=Path)
    experiment_audit=sub.add_parser('audit-experiment'); experiment_audit.add_argument('--manifest',type=Path,required=True); experiment_audit.add_argument('--trajectory',type=Path); experiment_audit.add_argument('--output',type=Path)
    f=sub.add_parser('verify-replay'); f.add_argument('--replay',type=Path,required=True)
    g=sub.add_parser('check-reproducibility'); g.add_argument('--left',type=Path,required=True); g.add_argument('--right',type=Path,required=True)
    args=p.parse_args()
    if args.cmd=='run':
        if not args.server_url: p.error('Runs require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        if args.max_steps is not None and args.max_steps < 1: p.error('--max-steps must be positive')
        if args.reward_config and (args.resume_run_id or args.restore_replay_id):
            p.error('--reward-config applies only to new runs; resumed runs retain their original rewards')
        directory=args.output or ROOT/'data'/'runs'
        try:
            generator_config=json.loads(args.generator_config.read_text(encoding='utf-8')) if args.generator_config else None
            reward_config=json.loads(args.reward_config.read_text(encoding='utf-8')) if args.reward_config else None
        except (OSError, json.JSONDecodeError) as error:
            p.error(str(error))
        if generator_config is not None and not isinstance(generator_config,dict): p.error('--generator-config must contain a JSON object')
        if args.reward_config and not isinstance(reward_config,dict): p.error('--reward-config must contain a JSON object')
        if reward_config is not None:
            try: reward_config=RewardConfig.model_validate(reward_config).model_dump(mode='json')
            except ValueError as error: p.error(f'--reward-config is invalid: {error}')
        if generator_config is not None and args.generated_world_seed is None: p.error('--generator-config requires --generated-world-seed')
        generated_world={"seed":args.generated_world_seed, **({"config":generator_config} if generator_config is not None else {})} if args.generated_world_seed is not None else None
        scenario_id=args.scenario or (None if generated_world else 'survival_room')
        manifest=ExperimentManifest(scenario_id=scenario_id or 'procedural',seed=args.seed,provider=args.provider,model=args.model,observation_mode=args.observation_mode,memory_mode=args.memory_mode,memory_window=args.memory_window,max_steps=args.max_steps,max_wall_seconds=args.max_wall_seconds,max_total_tokens=args.max_total_tokens,generator_version=1 if generated_world else None,generated_world=generated_world,reward_config=reward_config,agent_config={'include_research_snapshots':args.include_research_snapshots,'policy_state_mode':args.policy_state_mode})
        result=run_remote(provider_for(args.provider,args.seed,args.model),args.seed,args.server_url,args.memory_mode,args.max_steps,args.observation_mode,max_wall_seconds=args.max_wall_seconds,max_total_tokens=args.max_total_tokens,resume_run_id=args.resume_run_id,restore_replay_id=args.restore_replay_id,memory_window=args.memory_window,scenario_id=scenario_id,generated_world=generated_world,reward_config=reward_config,experiment_id=manifest.experiment_id,include_research_snapshots=args.include_research_snapshots,policy_state_mode=args.policy_state_mode)
        if result.provenance is not None:
            manifest=reconcile_manifest_provenance(manifest, result.provenance)
        elif result.world_manifest is not None:
            manifest=manifest.model_copy(update={"generated_world":result.world_manifest})
        manifest_path=manifest.persist(directory)
        jsonl=export_jsonl(result.records,directory/f'{result.run_id}.trajectory.jsonl')
        export_parquet([{**record,"observation":json.dumps(record["observation"]),"next_observation":json.dumps(record.get("next_observation")),"research_snapshot":json.dumps(record.get("research_snapshot")),"next_research_snapshot":json.dumps(record.get("next_research_snapshot")),"agent_context":json.dumps(record["agent_context"]),"agent_metadata":json.dumps(record.get("agent_metadata")),"events":json.dumps(record["events"]),"chosen_action":json.dumps(record["chosen_action"]),"metrics":json.dumps(record["metrics"])} for record in result.records],directory/f'{result.run_id}.parquet')
        print(json.dumps({"run_id":result.run_id,"experiment_id":manifest.experiment_id,"experiment_manifest":str(manifest_path),"outcome":result.terminal_reason,"stop_detail":result.stop_detail,"steps":result.steps,"events":sum(len(record["events"]) for record in result.records),"jsonl":str(jsonl)}))
    elif args.cmd=='benchmark':
        if not args.server_url: p.error('Benchmarks require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        print(json.dumps(benchmark_remote(args.runs,args.seed_start,args.output,args.server_url,args.provider,args.concurrency),indent=2))
    elif args.cmd=='benchmark-scale':
        if not args.server_url: p.error('Parallel benchmarks require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        try:
            workers=tuple(int(value) for value in args.workers.split(',') if value.strip())
            report=benchmark_parallel_scaling(args.runs,args.seed_start,args.output,args.server_url,args.provider,workers)
        except ValueError as error:
            p.error(str(error))
        print(json.dumps(report,indent=2))
    elif args.cmd=='generalize':
        if not args.server_url: p.error('Generalization evaluations require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        try:
            plan=GeneralizationPlan(SeedPartition.from_range('train',args.train_start,args.train_start+args.train_count),SeedPartition.from_range('validation',args.validation_start,args.validation_start+args.validation_count),SeedPartition.from_range('test',args.test_start,args.test_start+args.test_count))
            generator_config=json.loads(args.generator_config.read_text(encoding='utf-8')) if args.generator_config else None
            if generator_config is not None and not isinstance(generator_config,dict): raise ValueError('--generator-config must contain a JSON object')
            partition_paths={'train':args.train_generator_config,'validation':args.validation_generator_config,'test':args.test_generator_config}
            if args.generator_config and any(partition_paths.values()): raise ValueError('--generator-config cannot be combined with per-partition generator configs')
            if any(partition_paths.values()) and not all(partition_paths.values()): raise ValueError('provide train, validation, and test generator configs together')
            partition_configs={name:json.loads(path.read_text(encoding='utf-8')) for name,path in partition_paths.items()} if all(partition_paths.values()) else None
            if partition_configs is not None and any(not isinstance(config,dict) for config in partition_configs.values()): raise ValueError('per-partition generator configs must contain JSON objects')
            kwargs={}
            if generator_config is not None: kwargs['generator_config']=generator_config
            if partition_configs is not None: kwargs['generator_configs_by_partition']=partition_configs
            if args.observation_mode != 'normal': kwargs['observation_mode']=args.observation_mode
            report=evaluate_generalization_remote(plan,args.output,args.server_url,args.provider,args.concurrency,model=args.model,**kwargs)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        print(json.dumps(report,indent=2))
    elif args.cmd=='train-tabular':
        if not args.server_url: p.error('Tabular training requires --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        try:
            generator_config=json.loads(args.generator_config.read_text(encoding='utf-8')) if args.generator_config else None
            if generator_config is not None and not isinstance(generator_config,dict): raise ValueError('--generator-config must contain a JSON object')
            policy=TabularQPolicy(TabularQConfig(args.learning_rate,args.discount,args.epsilon))
            seeds=list(range(args.seed_start,args.seed_start+args.episodes))
            episodes=train_tabular_q(lambda: EmbodiedEnv(EmbodiedEnvConfig(server_url=args.server_url, observation_mode=args.observation_mode)),policy,seeds,args.max_steps,lambda seed: {"generated_world":{"seed":seed, **({"config":generator_config} if generator_config is not None else {})}})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        checkpoint=policy.save(args.checkpoint)
        checkpoint_hash=checkpoint_fingerprint(checkpoint)
        manifest=ExperimentManifest(scenario_id='procedural',seed=args.seed_start,provider='tabular_q',observation_mode=args.observation_mode,memory_mode='none',memory_window=1,max_steps=args.max_steps,generator_version=1,generated_world={"config":generator_config} if generator_config is not None else {},agent_config={"algorithm":"tabular_q","checkpoint":str(checkpoint),"checkpoint_fingerprint":checkpoint_hash,"episodes":args.episodes,"learning_rate":args.learning_rate,"discount":args.discount,"epsilon":args.epsilon,"phase":"train"})
        manifest_path=manifest.persist(checkpoint.parent)
        print(json.dumps({"policy":"tabular_q","observation_mode":args.observation_mode,"episodes":len(episodes),"checkpoint":str(checkpoint),"checkpoint_fingerprint":checkpoint_hash,"experiment_manifest":str(manifest_path),"mean_reward":sum(item["total_reward"] for item in episodes)/len(episodes),"terminal_reasons":{reason:sum(item["terminal_reason"]==reason for item in episodes) for reason in sorted({item["terminal_reason"] for item in episodes})}},indent=2))
    elif args.cmd=='evaluate-tabular':
        if not args.server_url: p.error('Tabular evaluation requires --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        try:
            generator_config=json.loads(args.generator_config.read_text(encoding='utf-8')) if args.generator_config else None
            if generator_config is not None and not isinstance(generator_config,dict): raise ValueError('--generator-config must contain a JSON object')
            policy=TabularQPolicy.load(args.checkpoint)
            checkpoint_hash=checkpoint_fingerprint(args.checkpoint)
            seeds=list(range(args.seed_start,args.seed_start+args.episodes))
            episodes=evaluate_tabular_q(lambda: EmbodiedEnv(EmbodiedEnvConfig(server_url=args.server_url, observation_mode=args.observation_mode)),policy,seeds,args.max_steps,lambda seed: {"generated_world":{"seed":seed, **({"config":generator_config} if generator_config is not None else {})}})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        manifest=ExperimentManifest(scenario_id='procedural',seed=args.seed_start,provider='tabular_q',observation_mode=args.observation_mode,memory_mode='none',memory_window=1,max_steps=args.max_steps,generator_version=1,generated_world={"config":generator_config} if generator_config is not None else {},agent_config={"algorithm":"tabular_q","checkpoint":str(args.checkpoint),"checkpoint_fingerprint":checkpoint_hash,"episodes":args.episodes,"phase":"evaluate"})
        manifest_path=manifest.persist(args.checkpoint.parent)
        print(json.dumps({"policy":"tabular_q","observation_mode":args.observation_mode,"checkpoint":str(args.checkpoint),"checkpoint_fingerprint":checkpoint_hash,"experiment_manifest":str(manifest_path),"episodes":episodes,"success_rate":sum(item["terminal_reason"]=="escaped" for item in episodes)/len(episodes),"mean_reward":sum(item["total_reward"] for item in episodes)/len(episodes)},indent=2))
    elif args.cmd=='generalize-tabular':
        if not args.server_url: p.error('Tabular generalization requires --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        try:
            generator_config=json.loads(args.generator_config.read_text(encoding='utf-8')) if args.generator_config else None
            if generator_config is not None and not isinstance(generator_config,dict): raise ValueError('--generator-config must contain a JSON object')
            partitions={'train':list(range(args.train_start,args.train_start+args.train_count)),'validation':list(range(args.validation_start,args.validation_start+args.validation_count)),'test':list(range(args.test_start,args.test_start+args.test_count))}
            policy=TabularQPolicy(TabularQConfig(args.learning_rate,args.discount,args.epsilon))
            factory=lambda: EmbodiedEnv(EmbodiedEnvConfig(server_url=args.server_url, observation_mode=args.observation_mode))
            options=lambda _partition, seed: {'generated_world': {'seed': seed, **({'config':generator_config} if generator_config is not None else {})}}
            train_tabular_q(factory,policy,partitions['train'],args.max_steps,lambda seed: options('train',seed))
            evaluated=evaluate_tabular_partitions(factory,policy,partitions,args.max_steps,options)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        checkpoint=policy.save(args.checkpoint); checkpoint_hash=checkpoint_fingerprint(checkpoint); args.output.mkdir(parents=True,exist_ok=True)
        config_fingerprint=generator_config_fingerprint(generator_config)
        rows=[{
            'partition':partition, 'outcome':episode['terminal_reason'], 'observation_mode':args.observation_mode,
            'generator_config':generator_config, 'generator_config_fingerprint':config_fingerprint,
            'hazard_kinds':generated_hazard_kinds(episode.get('world_manifest')),
            'room_count':generated_room_count(episode.get('world_manifest')),
            'mechanics_signature':generated_mechanics_signature(episode.get('world_manifest')),
            'world_validation':generated_world_validation(episode.get('world_manifest')),
            **episode,
        } for partition,episodes in evaluated.items() for episode in episodes]
        manifest=ExperimentManifest(scenario_id='procedural',seed=args.train_start,provider='tabular_q',observation_mode=args.observation_mode,memory_mode='none',memory_window=1,max_steps=args.max_steps,generator_version=1,generated_world={'config':generator_config} if generator_config is not None else {},world_distribution={name:tuple(seeds) for name,seeds in partitions.items()},agent_config={'algorithm':'tabular_q','checkpoint':str(checkpoint),'checkpoint_fingerprint':checkpoint_hash,'learning_rate':args.learning_rate,'discount':args.discount,'epsilon':args.epsilon,'phase':'train_validate_test'})
        manifest_path=manifest.persist(args.output)
        report=summarize_generalization(rows) | {'experiment_id':manifest.experiment_id,'experiment_manifest':manifest_path.name,'checkpoint':str(checkpoint),'checkpoint_fingerprint':checkpoint_hash,'engine_version':'rust-v1','observation_mode':args.observation_mode,'world_distribution':partitions,'generator_config':generator_config,'generator_configs_by_partition':None,'generator_config_fingerprint':config_fingerprint,'generator_config_fingerprints_by_partition':None,'episode_results':rows}
        (args.output/'tabular_generalization_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); export_jsonl(rows,args.output/'tabular_generalization_episodes.jsonl')
        print(json.dumps(report,indent=2))
    elif args.cmd=='analyze':
        records=load_jsonl(args.trajectory)
        report=summarize_trajectory(records)
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    elif args.cmd=='compare':
        report=compare_benchmarks(json.loads(args.left.read_text(encoding='utf-8')),json.loads(args.right.read_text(encoding='utf-8')))
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    elif args.cmd=='compare-generalization':
        try:
            report=compare_generalization_reports(json.loads(args.left.read_text(encoding='utf-8')),json.loads(args.right.read_text(encoding='utf-8')))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    elif args.cmd=='audit-generalization':
        try:
            source_report=json.loads(args.report.read_text(encoding='utf-8'))
            report=audit_generalization_report(source_report)
            if args.checkpoint is not None:
                expected=source_report.get('checkpoint_fingerprint')
                if not isinstance(expected,str) or not expected.startswith('sha256:'):
                    raise ValueError('generalization report does not declare a tabular checkpoint fingerprint')
                actual=checkpoint_fingerprint(args.checkpoint)
                if actual != expected:
                    raise ValueError('tabular checkpoint fingerprint does not match the generalization report')
                report['checkpoint_fingerprint_checked']=True
                report['checkpoint_fingerprint']=actual
            else:
                report['checkpoint_fingerprint_checked']=False
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    elif args.cmd=='filter':
        records=filter_trajectory(load_jsonl(args.trajectory),action_type=args.action_type,valid_only=args.valid_only,event_type=args.event_type)
        if args.csv: export_csv([{**record,"chosen_action":json.dumps(record.get("chosen_action")),"events":json.dumps(record.get("events")),"observation":json.dumps(record.get("observation")),"metrics":json.dumps(record.get("metrics"))} for record in records],args.output)
        else: export_jsonl(records,args.output)
        print(json.dumps({"records":len(records),"output":str(args.output)}))
    elif args.cmd=='audit-dataset':
        try:
            report=summarize_world_model_dataset(load_jsonl(args.trajectory))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    elif args.cmd=='audit-experiment':
        try:
            records=load_jsonl(args.trajectory) if args.trajectory else None
            report=audit_experiment_manifest(args.manifest,records)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
        if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    elif args.cmd=='verify-replay':
        print(json.dumps(verify_replay(json.loads(args.replay.read_text(encoding='utf-8'))),indent=2))
    elif args.cmd=='check-reproducibility':
        print(json.dumps(check_reproducibility(json.loads(args.left.read_text(encoding='utf-8')),json.loads(args.right.read_text(encoding='utf-8'))),indent=2))
if __name__=='__main__': main()
