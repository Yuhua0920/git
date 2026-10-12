# One narrowly scoped DACL grant; no owner, audit ACL, content or cache reset.
$ErrorActionPreference='Stop'
$taskProjectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskCacheRoot=Join-Path $taskProjectRoot '.pytest_cache'
$taskAuditRoot=Join-Path $taskProjectRoot 'outputs/t17a_permission_recovery'
$taskAclBefore=Get-Acl -LiteralPath $taskCacheRoot
& "$env:SystemRoot/System32/icacls.exe" $taskCacheRoot /grant 'ROG\yuhua:(OI)(CI)M' *> (Join-Path $taskAuditRoot 'administrator_acl_grant.txt')
if($LASTEXITCODE -ne 0){throw 'Minimal DACL grant failed'}
$taskItems=@(Get-Item -LiteralPath $taskCacheRoot)+@(Get-ChildItem -LiteralPath $taskCacheRoot -Force -Recurse)
$taskRecords=@($taskItems | ForEach-Object {
    $taskAcl=Get-Acl -LiteralPath $_.FullName
    [ordered]@{path=$_.FullName;owner=$taskAcl.Owner;inheritance_protected=$taskAcl.AreAccessRulesProtected
        sddl=$taskAcl.Sddl;entries=@($taskAcl.Access | ForEach-Object {
            [ordered]@{identity=$_.IdentityReference.Value;rights=$_.FileSystemRights.ToString()
                inheritance=$_.InheritanceFlags.ToString();propagation=$_.PropagationFlags.ToString()
                inherited=$_.IsInherited;type=$_.AccessControlType.ToString()}
        })}
})
$taskAclAfter=Get-Acl -LiteralPath $taskCacheRoot
if($taskAclBefore.Owner -ne $taskAclAfter.Owner -or
    $taskAclBefore.AreAccessRulesProtected -ne $taskAclAfter.AreAccessRulesProtected){throw 'Owner/protection drift'}
[ordered]@{cache_acl=$taskRecords;change='icacls root /grant ROG\yuhua:(OI)(CI)M'
    account=(whoami).Trim();owner_preserved=$true;inheritance_protection_preserved=$true} |
    ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $taskAuditRoot 'acl_after.json') -Encoding UTF8
'PASS' | Set-Content -LiteralPath (Join-Path $taskAuditRoot 'administrator_acl_grant_complete.txt') -Encoding UTF8
