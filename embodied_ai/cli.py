from __future__ import annotations
import argparse, json
from pathlib import Path
from .analysis import check_reproducibility, filter_trajectory, load_jsonl, summarize_trajectory, verify_replay
from .benchmark import GeneralizationPlan, SeedPartition, benchmark_remote, compare_benchmarks, evaluate_generalization_remote
from .datasets import export_csv, export_jsonl, export_parquet
from .experiments import ExperimentManifest
from .paths import ROOT
from .providers import CautiousProvider, ExplorerProvider, GeminiProvider, ScriptedProvider, RandomValidProvider, MockReasoningProvider
from .runner import run_remote

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
    a=sub.add_parser('run'); a.add_argument('--scenario',default='survival_room'); a.add_argument('--seed',type=int,default=42); a.add_argument('--provider',choices=provider_choices,default='scripted'); a.add_argument('--model'); a.add_argument('--max-steps',type=int); a.add_argument('--max-wall-seconds',type=float); a.add_argument('--max-total-tokens',type=int); resume=a.add_mutually_exclusive_group(); resume.add_argument('--resume-run-id'); resume.add_argument('--restore-replay-id'); a.add_argument('--observation-mode',choices=['minimal','normal','rich'],default='normal'); a.add_argument('--output',type=Path); a.add_argument('--memory-mode',choices=['none','recent'],default='recent'); a.add_argument('--server-url')
    a.add_argument('--memory-window',type=int,default=5,choices=range(1,101),metavar='1..100')
    b=sub.add_parser('benchmark'); b.add_argument('--scenario',default='survival_room'); b.add_argument('--provider',choices=provider_choices,default='scripted'); b.add_argument('--runs',type=int,default=5); b.add_argument('--seed-start',type=int,default=1000); b.add_argument('--concurrency',type=int,default=1); b.add_argument('--output',type=Path,default=ROOT/'data'/'exports'); b.add_argument('--server-url')
    generalize=sub.add_parser('generalize'); generalize.add_argument('--provider',choices=provider_choices,default='scripted'); generalize.add_argument('--train-start',type=int,default=0); generalize.add_argument('--train-count',type=int,default=5); generalize.add_argument('--validation-start',type=int,default=100); generalize.add_argument('--validation-count',type=int,default=2); generalize.add_argument('--test-start',type=int,default=200); generalize.add_argument('--test-count',type=int,default=2); generalize.add_argument('--concurrency',type=int,default=1); generalize.add_argument('--generator-config',type=Path); generalize.add_argument('--output',type=Path,default=ROOT/'data'/'exports'/'generalization'); generalize.add_argument('--server-url')
    c=sub.add_parser('analyze'); c.add_argument('--trajectory',type=Path,required=True); c.add_argument('--output',type=Path)
    d=sub.add_parser('compare'); d.add_argument('--left',type=Path,required=True); d.add_argument('--right',type=Path,required=True); d.add_argument('--output',type=Path)
    e=sub.add_parser('filter'); e.add_argument('--trajectory',type=Path,required=True); e.add_argument('--output',type=Path,required=True); e.add_argument('--action-type'); e.add_argument('--event-type'); e.add_argument('--valid-only',action='store_true'); e.add_argument('--csv',action='store_true')
    f=sub.add_parser('verify-replay'); f.add_argument('--replay',type=Path,required=True)
    g=sub.add_parser('check-reproducibility'); g.add_argument('--left',type=Path,required=True); g.add_argument('--right',type=Path,required=True)
    args=p.parse_args()
    if args.cmd=='run':
        if not args.server_url: p.error('Runs require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        if args.max_steps is not None and args.max_steps < 1: p.error('--max-steps must be positive')
        result=run_remote(provider_for(args.provider,args.seed,args.model),args.seed,args.server_url,args.memory_mode,args.max_steps,args.observation_mode,max_wall_seconds=args.max_wall_seconds,max_total_tokens=args.max_total_tokens,resume_run_id=args.resume_run_id,restore_replay_id=args.restore_replay_id,memory_window=args.memory_window,scenario_id=args.scenario)
        directory=args.output or ROOT/'data'/'runs'
        scenario_version=result.records[0].get("scenario_version") if result.records else None
        manifest=ExperimentManifest(scenario_id=args.scenario,scenario_version=scenario_version,seed=args.seed,provider=args.provider,model=args.model,observation_mode=args.observation_mode,memory_mode=args.memory_mode,memory_window=args.memory_window,max_steps=args.max_steps,max_wall_seconds=args.max_wall_seconds,max_total_tokens=args.max_total_tokens)
        manifest_path=manifest.persist(directory)
        jsonl=export_jsonl(result.records,directory/f'{result.run_id}.jsonl')
        export_parquet([{**record,"observation":json.dumps(record["observation"]),"agent_context":json.dumps(record["agent_context"]),"events":json.dumps(record["events"]),"chosen_action":json.dumps(record["chosen_action"]),"metrics":json.dumps(record["metrics"])} for record in result.records],directory/f'{result.run_id}.parquet')
        print(json.dumps({"run_id":result.run_id,"experiment_id":manifest.experiment_id,"experiment_manifest":str(manifest_path),"outcome":result.terminal_reason,"stop_detail":result.stop_detail,"steps":result.steps,"events":sum(len(record["events"]) for record in result.records),"jsonl":str(jsonl)}))
    elif args.cmd=='benchmark':
        if not args.server_url: p.error('Benchmarks require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        print(json.dumps(benchmark_remote(args.runs,args.seed_start,args.output,args.server_url,args.provider,args.concurrency),indent=2))
    elif args.cmd=='generalize':
        if not args.server_url: p.error('Generalization evaluations require --server-url for the authoritative Rust simulation. Start .\\scripts\\dev.ps1 first.')
        try:
            plan=GeneralizationPlan(SeedPartition.from_range('train',args.train_start,args.train_start+args.train_count),SeedPartition.from_range('validation',args.validation_start,args.validation_start+args.validation_count),SeedPartition.from_range('test',args.test_start,args.test_start+args.test_count))
            generator_config=json.loads(args.generator_config.read_text(encoding='utf-8')) if args.generator_config else None
            if generator_config is not None and not isinstance(generator_config,dict): raise ValueError('--generator-config must contain a JSON object')
            report=evaluate_generalization_remote(plan,args.output,args.server_url,args.provider,args.concurrency,generator_config)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(str(error))
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
    elif args.cmd=='filter':
        records=filter_trajectory(load_jsonl(args.trajectory),action_type=args.action_type,valid_only=args.valid_only,event_type=args.event_type)
        if args.csv: export_csv([{**record,"chosen_action":json.dumps(record.get("chosen_action")),"events":json.dumps(record.get("events")),"observation":json.dumps(record.get("observation")),"metrics":json.dumps(record.get("metrics"))} for record in records],args.output)
        else: export_jsonl(records,args.output)
        print(json.dumps({"records":len(records),"output":str(args.output)}))
    elif args.cmd=='verify-replay':
        print(json.dumps(verify_replay(json.loads(args.replay.read_text(encoding='utf-8'))),indent=2))
    elif args.cmd=='check-reproducibility':
        print(json.dumps(check_reproducibility(json.loads(args.left.read_text(encoding='utf-8')),json.loads(args.right.read_text(encoding='utf-8'))),indent=2))
if __name__=='__main__': main()
