import { TaskListPage } from '@/components/TaskListPage';
import { MODULES } from '@/theme/modules';
import { workflowTasks } from '@/mock/data';

export function WorkflowListPage() {
  return (
    <TaskListPage
      module={MODULES.workflow}
      tasks={workflowTasks}
      options={{ createText: '新建编排', sourceLabel: '主模型' }}
    />
  );
}
