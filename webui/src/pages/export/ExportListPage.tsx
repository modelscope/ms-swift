import { TaskListPage } from '@/components/TaskListPage';
import { MODULES } from '@/theme/modules';
import { exportTasks } from '@/mock/data';

export function ExportListPage() {
  return (
    <TaskListPage
      module={MODULES.export}
      tasks={exportTasks}
      options={{ createText: '新建导出', sourceLabel: '源模型 / ckpt' }}
    />
  );
}
