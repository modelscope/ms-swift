import { CopyButton } from './CopyText';

/**
 * 可复制的代码块。取代 antd 的 `<pre>` + Typography.Text copyable 那套手写组合。
 *
 * 部署详情页要放 curl 和 Python 两段调用示例，导出详情页要放输出目录，
 * 之前两个页面各自写了一份 inline style 的 pre，值还都是硬写的（圆角 11、字号 12.5）。
 * 合成一处之后换皮肤时只有这里要看。
 *
 * 复制按钮浮在右上角而不是排在标题旁边：代码块可能很宽要横向滚动，
 * 按钮跟着滚出视野就点不到了，所以钉在容器上而不是内容上。
 */
export function CodeBlock({ code, label }: { code: string; label?: string }) {
  return (
    <div className="relative">
      {/*
        pr-9 给右上角的复制按钮让位，否则长行会滑到按钮底下。
        whitespace-pre 而不是 pre-wrap：命令行示例折行之后就不能直接复制粘贴了，
        宁可横向滚动。
      */}
      <pre className="bg-muted text-foreground overflow-x-auto rounded-md px-3.5 py-3 pr-9 font-mono text-[12.5px] leading-5 whitespace-pre">
        {code}
      </pre>
      <span className="absolute top-2.5 right-2.5">
        <CopyButton text={code} label={label} />
      </span>
    </div>
  );
}

/**
 * 单行的路径 / 地址。跟 CodeBlock 同一个外观，但只有一行：
 * 过长时截断而不是滚动，复制按钮排在行尾而不是浮在角上。
 *
 * 单独一个组件而不是给 CodeBlock 加 oneLine 开关：两者的溢出处理是相反的
 * （截断 vs 滚动），塞进一个组件会变成两套互斥的类名靠布尔量切。
 */
export function CodeLine({ text, label }: { text: string; label?: string }) {
  return (
    <div className="bg-muted text-foreground flex items-center gap-2.5 rounded-md px-3.5 py-2.5 font-mono text-[12.5px]">
      <span className="min-w-0 flex-1 truncate">{text}</span>
      <CopyButton text={text} label={label} />
    </div>
  );
}
