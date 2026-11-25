# Path guard (`orkexa.fs`)

`orkexa.fs.SafeRoot` lets an agent's tools read files below one folder that a person granted, and nothing else.
It is product-neutral: Nodestro builds its folder grants and file tools on it (plan N-10), and any other host
can do the same. It uses only the standard library (`ctypes` on Windows and for Linux `openat2`).

```python
from orkexa.fs import FsError, SafeRoot, TextBudget

root = SafeRoot(
    "/home/ada/Documents/Offer 2026",
    protected=[product_data_dir],  # never a root, never reachable below one
    exclude=["**/*.key", "**/node_modules"],
    max_file_bytes=20 * 1024 * 1024,
)
grant_identity = root.identity  # store it with the grant

root.list("")  # [Entry(name, path, kind, size, ambiguous)]
root.read_bytes("notes/plan.md")
root.extract_text("specs/offer.pdf", budget=TextBudget(timeout=20, max_pages=200))

try:
    root.read_bytes("../.ssh/id_rsa")
except FsError as exc:
    exc.code  # "fs.parent_reference": map codes, never messages
```

Later, `SafeRoot(path, expect_identity=grant_identity)` refuses a folder that was swapped in under the same name
(`fs.root_changed`). `root.revoke()` refuses every later call and stops a read in progress at its next 1 MiB
chunk, a listing at its next entry and an extraction before it returns its text. `repr(root)` does not show
the path (it names a person's folders); `root.path` does.

A product with its own rules on paths below the root (a person's deny on one subfolder) passes
`on_open=check`. While a path is opened, `check(names)` gets the names the file system stored for the components
opened so far (`()` for the root) and raises to refuse. On a case-insensitive file system a request may open an
entry through another spelling (case, dotless i, ignorable characters), so compare the rule with these names,
not with the path asked for (and compare with `fold()`, which equates what such file systems may equate).

## How a path is checked

1. **Names first, the same on every system.** A path is relative, `/`-separated and normalized to NFC. Refused
   before anything is opened:
    * `..` (`fs.parent_reference`); absolute paths, drives (`C:`), UNC and device prefixes, a leading `~`
      (`fs.absolute_path`);
    * `\`, `:` (alternate data streams such as `file.txt:secret`), NUL and control characters, empty and `.`
      components, reserved device names (`CON`, `NUL.txt`, `COM1`, `LPT¹`), a trailing dot or space, 8.3 short
      names (`PROGRA~1`, `SECRET~1.TXT`), look-alike separators (fullwidth solidus, division slash, fraction
      slash), invalid UTF-8 (overlong encodings), more than 255 bytes per component, 4096 in all or more than
      `MAX_DEPTH` (128) components (`fs.invalid_path`).

    Nothing is decoded: `%2e%2e%2f` is a file name, not `../`. The Windows rules hold on Linux and macOS too, so a
    path never means a different file, or a device, on another system.
2. **Blocklist and patterns.** `blocked` (default `DEFAULT_BLOCKED`: `.ssh`, `.gnupg`, `.aws`, `.kube`,
   `.netrc`, `Library/Keychains`, the Windows credential folders and more) raises `fs.blocked`; `exclude` and
   `include` raise `fs.excluded`. Patterns use `*`, `?` and `**`, are anchored at the root unless they start
   with `**/`, and compare folded on every system (NFC, default-ignorable code points such as zero-width
   spaces removed, case-folded, dotless ı as i), so `.SSH`, `LIBRARY/KEYCHAINS`, `.ſsh`, `.s\u200bsh` and the
   NFD spelling of a name are blocked as well. A pattern that matches a folder covers everything below it. A
   blocked pattern that starts with `**/` is matched against the absolute path of the root plus the relative
   path, so `~/.config` as a root still blocks `gh/hosts.yml`. Patterns are matched in one pass as a set of
   positions (at most path components × pattern components comparisons, whatever the number of `**`), and a
   path has at most 128 components, so no path can make a check slow.
3. **The root, again.** Every call reopens the root and checks that it is the folder that was granted: the
   same identity (device and inode; volume serial number and file index on Windows) and, on macOS and Windows,
   the same final path. A root renamed and replaced by a link or another folder raises `fs.root_changed`.
4. **One component at a time, through descriptors.** No path string is resolved twice; the file is read from
   the very descriptor that passed the checks.

| System | How a component is opened | What is refused |
|---|---|---|
| Linux | `openat2(dirfd, name, RESOLVE_BENEATH \| RESOLVE_NO_SYMLINKS \| RESOLVE_NO_MAGICLINKS \| RESOLVE_NO_XDEV)` when the kernel offers it (5.6+; probed once per process, a later error of one path never switches to the fallback); otherwise `openat(dirfd, name, O_NOFOLLOW)` plus the device (`st_dev`) and the mount id (`/proc/self/fdinfo`) of every descriptor, which must be the root's (without `/proc` every path is refused). Folders with `O_DIRECTORY`; the last component of a file with `O_PATH` first, reopened through `/proc/self/fd` with `O_NONBLOCK \| O_NOCTTY` only once it is a regular file | symbolic links anywhere on the way (`fs.link`), magic links (`/proc/self/fd/N`), bind mounts (also of the same file system) and other file systems (`fs.cross_device`) |
| macOS | `openat(dirfd, name, O_NOFOLLOW)`, then `fcntl(F_GETPATH)` of every descriptor must lie below the root's (case-folded) | links, other volumes (`st_dev`), a path outside the root (`fs.outside_root`) |
| Windows | `CreateFileW("\\?\<root>\<a>\<b>", FILE_FLAG_OPEN_REPARSE_POINT \| FILE_FLAG_BACKUP_SEMANTICS)` per component, share mode without `FILE_SHARE_DELETE`; `GetFinalPathNameByHandleW` of every handle must be the root plus exactly the names asked for | symbolic links, junctions, volume mount points and other reparse points (`fs.link`), cloud placeholders (`fs.cloud_file`), a final path elsewhere (`fs.outside_root`) |

On Windows there is no documented `openat`, so the walk pins the path instead: a folder whose handle is open
without `FILE_SHARE_DELETE` cannot be renamed, deleted or replaced, so the next component resolves through
exactly the folders already checked. `FILE_FLAG_OPEN_REPARSE_POINT` alone protects only the last component (the
W0-6 spike showed that a folder link before it is followed), which is why every component is opened and checked
on its own, and why the final path of each handle is the binding check. `\\?\` turns off Win32 name rewriting
(trailing dots, device names); `FILE_NAME_NORMALIZED` expands 8.3 names. Nothing needs administrator rights.

Links are never followed, also links that stay inside the root; a listing shows them as `kind="link"`.

On POSIX only the descriptor of the current folder stays open (the parent's is closed once the next component is
open), so a deep path does not use up descriptors. On Linux the last component of a file is opened with `O_PATH`,
which opens nothing (no FIFO, socket or device driver is touched), and reopened through `/proc/self/fd` only once
`fstat` of that descriptor shows a regular file or a folder; the reopen names that very inode, so no swap can
come in between (review N7). This is used only when `/proc` is the real procfs (`statfs` magic), checked once
per process. On macOS and on Linux without `/proc`, the last component is looked at with
`fstatat` first and opened with `O_NONBLOCK | O_NOCTTY`, and the opened descriptor must be the entry that was
looked at; there a special file swapped in between the two calls is opened (without blocking) before it is
refused.

**The name the file system answered with.** On case-insensitive file systems a name can open an entry whose
stored name differs by more than the folding above (HFS+ ignores some code points; the upcase tables of exFAT and
NTFS and the casefold of ext4 differ from Unicode case folding). So the stored name of every opened entry must
fold to the name asked for, or the path is refused (`fs.ambiguous_name`): Windows compares the final path of
each handle; macOS reads the stored name of the descriptor (`fgetattrlist`, `ATTR_CMN_NAME`); Linux scans the
folder for the entry with that identity on every file system that is not known to compare names byte for byte
(only ext2/3/4, Btrfs, XFS, tmpfs, F2FS, overlayfs, ramfs, bcachefs, squashfs, EROFS, NILFS and ReiserFS are;
so FAT, exFAT, NTFS, HFS+, SMB, NFS, 9p, FUSE and virtiofs, ISO 9660, ZFS and unknown ones are scanned) and in
folders with the casefold attribute (`fstatfs`, `FS_IOC_GETFLAGS`). Case differences alone are fine. CI checks
this on a real ext4 casefold folder (a loop-mounted image in the `fs-fallbacks` job).

## Files that are refused

* **Special files**: FIFOs, sockets and devices (`fs.not_regular_file`), without blocking on a FIFO.
* **Hard links** (decision A-15): a file with `st_nlink > 1` (Windows `nNumberOfLinks`) is refused with
  `fs.hard_link`, because it may be the very file of another folder and no path shows that; `stat` refuses it too
  and `list` shows it with size 0. `allow_hard_links=True` opts in per root (for example a backup folder).
  Folders are not affected.
* **Size**: a file above `max_file_bytes` (default 50 MiB) is refused from its size alone, before a byte is read
  (`fs.too_large`, "The file has N bytes; at most M bytes can be read"); a file that grows while it is read is
  cut off at the limit and refused too.
* **Listings** read at most `max_entries` entries (default 10,000; hidden and excluded ones count too) within
  `max_list_seconds` (default 10), entry by entry; beyond either, `fs.too_many_entries` with `reason`
  `entries` or `timeout`. Finding a name stored in another Unicode form (below) reads at most 100,000 entries
  of a folder.

## Unicode and case

Input paths are normalized to NFC. A name stored in another form (NFD, as macOS and some tools write it) is found
when exactly one entry of the folder has that NFC form; when several have it, the exact name opens and any other
spelling raises `fs.ambiguous_name`. Listings mark such entries `ambiguous=True`. The blocklist, the patterns and
the protected-folder names compare NFC plus case-folded, which fails closed on case-insensitive file systems
(APFS, NTFS); protected folders are also compared by identity, so no spelling reaches them. Bidirectional
controls and zero-width characters stay readable (P0-2 decision) and `Entry.display` shows them escaped
(`invoice\u202efdp.exe`).

## Roots that cannot be granted

`SafeRoot` refuses, with `fs.protected_root`:

* a file-system or drive root (`/`, `C:\`);
* the home folder itself and every folder that contains it (`home=`, default: the home folder of the process's
  user), the folders that hold all users' home folders (`/home`, `/Users`, `/var/home`, `/export/home`,
  `/usr/home`, `C:\Users`, the parent of `%USERPROFILE%` and `%PUBLIC%`), and every folder directly inside them:
  any user's home folder (`/home/bob`, `/Users/Shared`, `C:\Users\Public`), also when the process runs as another
  user (a service account) whose home is elsewhere (review N1). A folder *inside* a home folder can be granted;
* a folder that a blocked pattern starting with `**/` covers by its absolute path (`~/.ssh`,
  `~/Library/Keychains` or a folder inside one): by its canonical path, by the path as it was given, and by
  every path its links lead through as they are resolved one at a time, so a `~/.ssh` that is a link to
  `/vault/keys`, or a harmless-looking link to `~/.ssh`, is refused (review N2). The chain of links must end at
  the very folder that was resolved (same identity), so a link flipped between resolving the root and reading
  its names is refused too. A folder granted under its own name only (`/vault/keys` itself) cannot be
  recognized;
* a system folder or a folder inside one: the list of nodestro-api's `output_folder.py` (`/bin`, `/etc`,
  `/usr`, `/var`, `/System`, `/Library`, `/private` ... with the exceptions `/var/home`, `/var/mnt`,
  `/var/tmp`, `/run/media`, `/private/tmp`, `/private/var/folders`) plus `/root`, and on Windows the folders
  that `SystemRoot`, `windir`, `ProgramFiles`, `ProgramFiles(x86)`, `ProgramW6432` and `ProgramData` name
  (`C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`, `C:\ProgramData` when they are not set);
* a `protected` folder (the product's data folder) or a folder inside one.

Below a root, protected folders are refused by name (folded, also before they exist: a missing protected folder
is refused, not skipped) and by identity. Identities are learned when the root is created (for protected folders
that exist then) and again on every call, and every identity seen is kept ("never forget"), so a protected folder
created, deleted and recreated, or moved within the root after the grant is still refused once any call has seen
it. Two consequences, both documented rather than changed (review N4): a protected folder that is created after
the grant and moved within the root before any call has seen it is reachable under its new name; and a folder
that later gets the inode number of a deleted protected folder is refused (fails closed, availability only).

Folders are compared component by component (never by string prefix), NFC plus case-folded, and by identity, so
`/library`, a firmlink detour (`/System/Volumes/Data/...`) or a link to the data folder is refused too. A root
that *contains* the data folder (a home folder subfolder, say) is allowed; the data folder is then blocked below
it (`fs.blocked`).

Network paths raise `fs.network_path`: UNC shares (`\\server\share`), also administrative shares of this computer
(`\\localhost\C$`, `\\<computer name>\C$`, use `C:\` instead), `\\?\UNC\...`, mapped network drives
(`GetDriveTypeW` is `DRIVE_REMOTE`) and roots whose final path is a share. Windows device paths (`\\.\`, `\??\`,
`\\?\GLOBALROOT`, `\\?\Volume{...}`) raise `fs.invalid_path`; `\\?\C:\x` is `C:\x`.

**OneDrive** files on demand are cloud placeholders (reparse tag `IO_REPARSE_TAG_CLOUD*`, or the recall
attributes); reading one downloads it, so it is refused with `fs.cloud_file` unless the root is created with
`allow_cloud_files=True`. Then the file is opened again without `FILE_FLAG_OPEN_REPARSE_POINT` and must be the
same file (identity).

### Consistency with nodestro-runtime's access grants

N-06 (`tool_policy.folder_parts`) compares folder *names* and expects callers to resolve links first; `orkexa.fs`
is the file-system guard behind it. Both use NFC, whole components, case-folded comparison for denies and system
folders, the same system-folder environment variables, `\\?\C:\x` as `C:\x`, and refuse Windows device paths,
`:` in a component and 8.3 names in input. Differences, all failing closed: `SafeRoot` canonicalizes the root
(links resolved, 8.3 names expanded by the final path) where N-06 refuses a short name in a grant path; it refuses
every UNC root, where N-06 maps `\\localhost\C$` to drive `C:` for comparison; and it refuses relative names with a
trailing dot or space, where N-06 drops them as Windows does.

## Creating and renaming

With `writable=True`, `create_file(path, data)` creates a new file and `rename(source, target)` renames a file or
folder. Neither ever replaces an existing entry (`fs.exists`) or follows a link:

* POSIX: `openat(dirfd, name, O_CREAT | O_EXCL | O_NOFOLLOW)` in the checked folder; `renameat2(RENAME_NOREPLACE)`
  on Linux and `renameatx_np(RENAME_EXCL)` on macOS between the checked folders' descriptors. Without either (old
  kernels or file systems), a file is hard-linked to the new name, which fails if it exists, the old name is
  checked to still be that file and unlinked; if anything fails the new name is removed again (`FsError`, never a
  second link left behind). A folder cannot be renamed then (`fs.unsupported_type`).
* Windows: `CREATE_NEW` below the held folder handles, then the final path of the new handle;
  `MoveFileExW` without `MOVEFILE_REPLACE_EXISTING` and without `MOVEFILE_COPY_ALLOWED`.

Links, protected folders, the folders above a protected folder and files with several hard links are not
renamed, and nothing is renamed *to* a protected folder or a folder above one, also while the protected folder
does not exist yet (review N3), so nothing can be planted where the product will create its data folder. A rename never changes what may be read: a file must pass the blocklist and the patterns under its old
and its new name (so a file outside `include` cannot be moved into it), and a folder may only get a name under
which every blocked, exclude and include pattern decides the same for everything below it (the pattern states
after both names are compared), else `fs.excluded`. A root is read only by default (`fs.read_only`).

## Text extraction (PS-1)

`extract_text(path, budget=TextBudget(...))` reads a text file (`TEXT_SUFFIXES`: UTF-8, or UTF-16 with a byte
order mark; NUL bytes mean binary, `fs.unsupported_type`) or a PDF through the guard. PDFs go through
`orkexa.retrieval.pdf_pages_isolated`: a separate interpreter with a minimal environment that is killed at
`budget.timeout`, with `max_pages`, `max_chars` and `max_memory_mb`. The memory budget is `RLIMIT_AS` on Linux,
a job object with a process memory limit on Windows, and on macOS (which ignores `RLIMIT_AS`) a watcher that
kills the process within about 20 ms of its physical footprint (`proc_pid_rusage`: resident plus compressed and
swapped-out memory, review N5) passing the budget. Where the budget cannot be enforced, a PDF is refused
(`fs.unsupported_type`) rather than read without it: on any other system, on macOS when the footprint of a
process cannot be measured (also when it stops being measurable while the worker runs: the worker is killed),
and on Windows when the job object cannot be created or the worker cannot be put into it (review N6; directly,
`pdf_pages_isolated` raises `RetrievalError` with `reason` `unenforced`). The same worker launcher, with the same
memory budget on Linux, macOS and Windows, runs `orkexa.web.fetch.html_to_text_isolated`. A budget hit raises
`fs.too_complex` with `reason` `pages`, `chars`, `timeout`, `memory`, `limit` or `crash`; a PDF bomb (a page
that keeps the parser busy) ends within the time budget plus a few seconds of process start and kill. Office
formats are not part of Orkexa (Nodestro's runtime has its own DOCX and XLSX readers).

## Error codes

`FS_ERROR_CODES` maps every code to its meaning: `fs.invalid_path`, `fs.absolute_path`, `fs.parent_reference`,
`fs.link`, `fs.outside_root`, `fs.cross_device`, `fs.hard_link`, `fs.not_regular_file`, `fs.not_a_directory`,
`fs.is_a_directory`, `fs.not_found`, `fs.ambiguous_name`, `fs.blocked`, `fs.excluded`, `fs.too_large`,
`fs.too_many_entries`, `fs.protected_root`, `fs.network_path`, `fs.cloud_file`, `fs.root_changed`, `fs.revoked`,
`fs.read_only`, `fs.exists`, `fs.permission`, `fs.unreadable`, `fs.unsupported_type`, `fs.too_complex`.
`FsError` is a `PermissionError`; its message never contains an absolute path.

## The attack catalog

`orkexa.testing.pathguard.CATALOG` holds every N-10 case of the P0-2 attack catalog (`N10-01` to `N10-23`) and
every gap of the W0-6 spike (`W06-1` to `W06-5`), with the platforms where each is physically possible and the
codes that refuse it. Orkexa's tests (`tests/fs`) run all of them on Linux, macOS and Windows in CI. A product
runs the portable part against its own tools:

```python
from orkexa.testing import pathguard

layout = pathguard.build_layout(tmp_path)  # root/, outside/ with a canary, links where possible
assert pathguard.check_reader(my_read_tool, layout, refusal=MyToolError) == []
assert pathguard.swap_race(my_read_tool, layout, iterations=10_000) == []
```

`check_reader` first checks a positive control (the files inside the root must be returned), then requires every
attack to be refused with the expected type (`refusal`, default `FsError`) and code (`code_of`, default the
`code` attribute); a crash is a finding. `swap_race` reports a run in which fewer than `min_swaps` swaps happened
or the reader never returned the regular file, so a race that did not run cannot pass.

## Remaining risks

* **The guard reads as the user.** It decides which path may be opened; it is no sandbox. Code running in the
  same process can bypass it.
* **Network file systems on Linux and macOS** (NFS, SMB mounts) are refused only when they are mounted *below*
  a root (`fs.cross_device`); a root *on* such a mount is treated like a local folder, and the server decides
  what a name means.
* **Windows renames** go through `MoveFileExW` with a path below held folder handles: the source entry itself is
  not pinned, so a concurrent swap inside the root can rename a different entry of the same folder (never one
  outside, never through a link). Kept as documented (review L8).
* **Linux without `openat2` and without `/proc`**: the fallback cannot see same-file-system bind mounts, so it
  refuses every path (`fs.cross_device`) rather than guess.
* **macOS memory budget** is polled every 20 ms: a PDF that allocates very fast can exceed it briefly before
  the process is killed.
* **Special files on macOS** (and Linux without `/proc`): without `O_PATH`, a FIFO or device swapped in between
  `fstatat` and the open is opened (non-blocking, no controlling terminal) before it is refused. Creating a
  device node needs root.
* **Protected folders** are recognized by identity once the guard has seen them (at the grant, or any call
  after they were created): one created after the grant and moved within the root before any call is reachable
  under its new name (review N4, documented).
* **A blocked folder granted under its own name** (the target of a `~/.ssh` link, given directly) is not
  recognized; only names on the way to the root are.
* **Hard links opted in** (`allow_hard_links=True`) may expose a file that also lives elsewhere; that is the
  operator's choice.
* **Bidirectional and zero-width characters** are readable; showing names escaped (`Entry.display`) is the
  caller's job.
* The independent review of #109 is fixed (each finding has a regression test in `tests/fs/test_hardening.py`),
  and so is its re-review (N1-N3, N5-N7 fixed, N4 documented; `tests/fs/test_rereview.py`); G-5 is due before
  the guard protects real user data (N-10 acceptance, ORK-045).
