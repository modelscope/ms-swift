import { TaskListPage } from '@/components/TaskListPage';
import { MODULES } from '@/theme/modules';
import { deployTasks } from '@/mock/data';

export function DeployListPage() {
  return (
    <TaskListPage
      module={MODULES.deploy}
      tasks={deployTasks}
      options={{ createText: '新建部署', sourceLabel: '模型', showEndpoint: true }}
    />
  );
}
