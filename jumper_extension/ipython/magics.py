import logging

from IPython.core.magic import Magics, line_magic, magics_class

from jumper_extension.ipython.utilities import is_pure_line_magic_cell, get_called_line_magics
from jumper_extension.core.service import PerfmonitorMagicAdapter

logger = logging.getLogger("extension")


@magics_class
class PerfmonitorMagics(Magics):
    def __init__(self, shell, magic_adapter: PerfmonitorMagicAdapter):
        super().__init__(shell)
        self.magic_adapter = magic_adapter

    def pre_run_cell(self, info):
        raw_cell = info.raw_cell
        called_line_magics = get_called_line_magics(raw_cell)
        should_skip_report = is_pure_line_magic_cell(raw_cell)
        self.magic_adapter.on_pre_run_cell(
            raw_cell,
            called_line_magics,
            should_skip_report,
        )

    def post_run_cell(self, result):
        self.magic_adapter.on_post_run_cell(result.result)

    @line_magic
    def perfmonitor_resources(self, line):
        """Display available hardware resources (CPUs, memory, GPUs)"""
        self.magic_adapter.perfmonitor_resources(line)

    @line_magic
    def perfmonitor_start(self, line):
        """Start performance monitoring with specified interval
        (default: 1 second)"""
        self.magic_adapter.perfmonitor_start(line)

    @line_magic
    def perfmonitor_stop(self, line):
        """Stop the active performance monitoring session"""
        self.magic_adapter.perfmonitor_stop(line)

    @line_magic
    def perfmonitor_plot(self, line):
        """Open interactive plot with widgets for exploring performance data"""
        self.magic_adapter.perfmonitor_plot(line)

    @line_magic
    def perfmonitor_enable_perfreports(self, line):
        """Enable automatic performance reports after each cell execution"""
        self.magic_adapter.perfmonitor_enable_perfreports(line)


    @line_magic
    def perfmonitor_disable_perfreports(self, line):
        """Disable automatic performance reports after cell execution"""
        self.magic_adapter.perfmonitor_disable_perfreports(line)

    @line_magic
    def perfmonitor_perfreport(self, line):
        """Show performance report with optional cell range and level
        filters"""
        self.magic_adapter.perfmonitor_perfreport(line)

    @line_magic
    def perfmonitor_export_perfdata(self, line):
        """Export performance data or push as DataFrame

        Usage:
          %perfmonitor_export_perfdata --file <path> [--level LEVEL]
            # export to file
          %perfmonitor_export_perfdata [--level LEVEL]
            # push DataFrame
        """
        perfdata = self.magic_adapter.perfmonitor_export_perfdata(line)
        self.shell.push(perfdata)

    @line_magic
    def perfmonitor_export_cell_history(self, line):
        """Export cell history or push as DataFrame

        Usage:
          %perfmonitor_export_cell_history --file <path>  # export to file
          %perfmonitor_export_cell_history                # push DataFrame
        """
        cell_history_data = self.magic_adapter.perfmonitor_export_cell_history(line)
        self.shell.push(cell_history_data)

    @line_magic
    def perfmonitor_load_perfdata(self, line):
        """Import performance data

        Usage:
          %perfmonitor_load_perfdata --file <path>
            # import from file
        """
        perfdata = self.magic_adapter.perfmonitor_load_perfdata(line)
        self.shell.push(perfdata)

    @line_magic
    def perfmonitor_load_cell_history(self, line):
        """Import cell history or push as DataFrame

        Usage:
          %perfmonitor_load_cell_history --file <path>  # import from file
        """
        cell_history_data = self.magic_adapter.perfmonitor_load_cell_history(line)
        self.shell.push(cell_history_data)

    @line_magic
    def export_session(self, line):
        """Export full session (CSV+manifest) to directory or zip.

        Usage:
          %export_session
          %export_session my_dir
          %export_session my_archive.zip
        """
        self.magic_adapter.export_session(line)

    @line_magic
    def import_session(self, line):
        """Import full session (CSV+manifest) from directory or zip.

        Usage:
          %import_session path/to/dir-or-zip
        """
        self.magic_adapter.import_session(line)

    @line_magic
    def perfmonitor_fast_setup(self, line):
        """Quick setup: enable ipympl interactive plots, start perfmonitor, and enable perfreports"""
        # Enable ipympl interactive plots
        try:
            self.shell.run_line_magic('matplotlib', 'ipympl')
            print("[JUmPER]: Enabled ipympl interactive plots")
        except Exception as e:
            logger.warning(f"Failed to enable ipympl interactive plots: {e}")
        self.magic_adapter.perfmonitor_fast_setup(line)

    @line_magic
    def show_cell_history(self, line):
        """Show interactive table of all executed cells with timestamps"""
        self.magic_adapter.show_cell_history(line)

    @line_magic
    def perfmonitor_help(self, line):
        """Show comprehensive help information for all available commands"""
        self.magic_adapter.perfmonitor_help(line)

    @line_magic
    def start_write_script(self, line):
        """
        Start recording code from cells to a Python script.

        Usage:
          %start_write_script [output_path]

        Examples:
          %start_write_script
          %start_write_script my_script.py
        """
        self.magic_adapter.start_write_script(line)

    @line_magic
    def end_write_script(self, line):
        """
        Stop recording and save accumulated code to file.

        Usage:
          %end_write_script
        """
        self.magic_adapter.end_write_script(line)
