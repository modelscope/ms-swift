import { TaskListPage } from '@/components/TaskListPage';
import { MODULES } from '@/theme/modules';
import { evalTasks } from '@/mock/data';

export function EvalListPage() {
  return (
    <TaskListPage
      module={MODULES.eval}
      tasks={evalTasks}
      options={{ createText: '新建评测', sourceLabel: '被测模型 / ckpt' }}
    />
  );
}
