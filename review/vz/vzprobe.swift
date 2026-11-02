// Feasibility probe (review only): can this Mac boot a minimal Linux guest with Virtualization.framework,
// with no network device and only a vsock and a serial console? Prints timings and the guest's console.
import Foundation
import Virtualization

setvbuf(stdout, nil, _IONBF, 0)
let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: vzprobe <kernel> <initrd> [seconds]"); exit(2) }
let seconds = args.count > 3 ? Double(args[3]) ?? 30 : 30
print("isSupported=\(VZVirtualMachine.isSupported)")
let cfg = VZVirtualMachineConfiguration()
cfg.cpuCount = 1
cfg.memorySize = 256 * 1024 * 1024
let boot = VZLinuxBootLoader(kernelURL: URL(fileURLWithPath: args[1]))
boot.initialRamdiskURL = URL(fileURLWithPath: args[2])
boot.commandLine = "console=hvc0 rdinit=/bin/sh quiet"
cfg.bootLoader = boot
let serial = VZVirtioConsoleDeviceSerialPortConfiguration()
let outPipe = Pipe()
let inPipe = Pipe()
serial.attachment = VZFileHandleSerialPortAttachment(fileHandleForReading: inPipe.fileHandleForReading,
                                                    fileHandleForWriting: outPipe.fileHandleForWriting)
cfg.serialPorts = [serial]
cfg.socketDevices = [VZVirtioSocketDeviceConfiguration()]
cfg.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]
do { try cfg.validate(); print("validate=ok") } catch { print("validate=FAILED \(error)"); exit(3) }
var console = ""
outPipe.fileHandleForReading.readabilityHandler = { h in
    let d = h.availableData
    if let s = String(data: d, encoding: .utf8) { console += s; FileHandle.standardOutput.write(d) }
}
let queue = DispatchQueue(label: "vm")
let vm = VZVirtualMachine(configuration: cfg, queue: queue)
let t0 = Date()
queue.async {
    vm.start { result in
        switch result {
        case .success: print("\nstarted after \(Date().timeIntervalSince(t0))s")
            queue.asyncAfter(deadline: .now() + 3) {
                inPipe.fileHandleForWriting.write("echo GUEST_OK $(uname -m) $(cat /proc/cmdline); ls /sys/class/net\n".data(using: .utf8)!)
            }
        case .failure(let e): print("start=FAILED \(e)"); exit(4)
        }
    }
}
RunLoop.main.run(until: Date().addingTimeInterval(seconds))
print("\nconsole_has_guest_ok=\(console.contains("GUEST_OK"))")
exit(console.contains("GUEST_OK") ? 0 : 5)
