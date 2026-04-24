"""
MIVA Studio command-line interface.

Provides CLI for enrollment, generation, evaluation, and monitoring.
"""

import click
import logging
from pathlib import Path
from miva.config import get_config
from miva.pipeline import MIVAPipeline
from miva.observability import SessionTracer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MIVA Studio — Production RAG for Identity-Critical Visual Generation"""
    pass


@cli.command()
@click.option('--subject_id', required=True, help='Subject identifier')
@click.option('--prompt', default='professional portrait', help='Generation prompt')
@click.option('--num_outputs', default=1, type=int, help='Number of images to generate')
@click.option('--output_dir', default='./outputs', help='Output directory')
def generate(subject_id: str, prompt: str, num_outputs: int, output_dir: str):
    """Generate identity-consistent images for a subject."""
    click.echo(f"\n{click.style('Generating images...', fg='blue', bold=True)}")
    
    pipeline = MIVAPipeline()
    results = pipeline.generate(
        subject_id=subject_id,
        prompt=prompt,
        num_outputs=num_outputs,
        output_dir=output_dir
    )
    
    click.echo(f"\n{click.style('Results:', fg='green', bold=True)}")
    for result in results:
        if result.success:
            click.echo(f"  ✓ {result.session_id}")
            click.echo(f"    Output: {result.output_path}")
            click.echo(f"    Identity score: {result.final_identity_score:.4f}")
            click.echo(f"    Attempts: {result.attempts}")
        else:
            click.echo(f"  ✗ {result.session_id}")
            click.echo(f"    Reason: {result.failure_reason}")


@cli.command()
def health_check():
    """Run system health check."""
    click.echo(f"\n{click.style('Health Check', fg='blue', bold=True)}")
    
    pipeline = MIVAPipeline()
    if pipeline.health_check():
        click.echo(f"{click.style('✓ All checks passed', fg='green')}\n")
    else:
        click.echo(f"{click.style('✗ Some checks failed', fg='red')}\n")


@cli.command()
def list_sessions():
    """List recent generation sessions."""
    config = get_config()
    tracer = SessionTracer(config)
    
    traces = tracer.list_traces(limit=10)
    
    click.echo(f"\n{click.style('Recent Sessions:', fg='blue', bold=True)}")
    for trace in traces:
        summary = tracer.get_trace_summary(trace)
        status = click.style('✓', fg='green') if summary['success'] else click.style('✗', fg='red')
        click.echo(f"\n{status} {summary['session_id'][:8]}... ({summary['subject_id']})")
        click.echo(f"   Attempts: {summary['attempts']}, Score: {summary['identity_score']}")


@cli.command()
@click.option('--session_id', required=True, help='Session ID to inspect')
def inspect_session(session_id: str):
    """Inspect a specific session trace."""
    config = get_config()
    tracer = SessionTracer(config)
    
    trace = tracer.load_trace(session_id)
    if not trace:
        click.echo(f"{click.style('Session not found', fg='red')}")
        return
    
    summary = tracer.get_trace_summary(trace)
    
    click.echo(f"\n{click.style('Session Trace', fg='blue', bold=True)}")
    for key, value in summary.items():
        click.echo(f"  {key}: {value}")
    
    click.echo(f"\n{click.style('Attempts:', fg='blue')}")
    for attempt in trace.attempts:
        click.echo(
            f"  {attempt.attempt_number}: {attempt.guardrail_decision} "
            f"(identity={attempt.identity_score}, quality={attempt.artifact_score})"
        )


@cli.command()
def show_config():
    """Show current configuration."""
    config = get_config()
    click.echo(config.summary())


def main():
    """Entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
