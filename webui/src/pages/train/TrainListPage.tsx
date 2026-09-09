import { TaskListPage } from '@/components/TaskListPage';
import { MODULES } from '@/theme/modules';
import { trainTasks } from '@/mock/data';

export function TrainListPage() {
  return (
    <TaskListPage
      module={MODULES.train}
      tasks={trainTasks}
      options={{
        createText: '新建训练',
        sourceLabel: '基座模型',
        showProgress: true,
        showMetrics: true,
      }}
    />
  );
}
